"""End-to-end tests of the engine audio API through Flask's test client."""
import io
import os

from .conftest import audio

ANALYZE = "/api/v1/analyze"
MULTIPART = "multipart/form-data"

HEALTHY_START = "healthy/CAG 5756_Start.wav"
SHORT_START = "healthy/CAD 1530_Start.wav"      # a real 1.9 s engine start
HEALTHY_IDLE = "healthy/CAQ 1654_Idle.wav"
HEALTHY_REVS = "healthy/CAL 2467_High RPM.wav"
KNOCKING = "knocking/Engine knock 6.wav"


def post(client, data, headers):
    return client.post(ANALYZE, data=data, headers=headers, content_type=MULTIPART)


def three_stages(start=HEALTHY_START, idle=HEALTHY_IDLE, revs=HEALTHY_REVS):
    return {"audio_start": audio(start), "audio_idle": audio(idle), "audio_acceleration": audio(revs)}


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.get_json()["models_loaded"] is True


def test_missing_token(client):
    assert post(client, {}, {}).status_code == 401


def test_wrong_token(client):
    assert post(client, {}, {"Authorization": "Bearer wrong"}).status_code == 401


def test_no_file(client, auth):
    assert post(client, {}, auth).status_code == 400


def test_unsupported_format(client, auth):
    r = post(client, {"audio_idle": (io.BytesIO(b"hello" * 5000), "notes.txt")}, auth)
    assert r.status_code == 422
    assert r.get_json()["stages"]["audio_idle"]["code"] == "BAD_FORMAT"


def test_file_too_small(client, auth):
    r = post(client, {"audio_idle": (io.BytesIO(b"RIFF" + b"0" * 500), "tiny.wav")}, auth)
    assert r.status_code == 422
    assert r.get_json()["stages"]["audio_idle"]["code"] == "TOO_SMALL"


def test_unreadable_file(client, auth):
    r = post(client, {"audio_idle": (io.BytesIO(os.urandom(20000)), "junk.wav")}, auth)
    assert r.status_code == 422
    assert r.get_json()["stages"]["audio_idle"]["code"] == "UNREADABLE"


def test_three_healthy_stages(client, auth):
    r = post(client, {**three_stages(), "session_id": "S-001"}, auth)
    body = r.get_json()
    assert r.status_code == 200
    assert body["fault_class"] == "healthy"
    assert body["color_indicator"] == "green"
    assert body["session_id"] == "S-001"
    assert set(body["stages"]) == {"audio_start", "audio_idle", "audio_acceleration"}


def test_worst_stage_decides_the_result(client, auth):
    r = post(client, three_stages(idle=KNOCKING), auth)
    body = r.get_json()
    assert r.status_code == 200
    assert body["fault_class"] == "knocking"
    assert body["color_indicator"] == "red"
    assert body["mhs_score"] == min(s["mhs_score"] for s in body["stages"].values())


def test_short_engine_start_is_accepted(client, auth):
    r = post(client, three_stages(start=SHORT_START), auth)
    assert r.status_code == 200
    assert r.get_json()["stages"]["audio_start"]["valid"] is True


def test_short_clip_rejected_outside_start_stage(client, auth):
    r = post(client, three_stages(idle=SHORT_START), auth)
    assert r.status_code == 422
    assert r.get_json()["stages"]["audio_idle"]["code"] == "TOO_SHORT"


def test_tone_is_not_an_engine(client, auth, synthetic_audio):
    data = three_stages()
    data["audio_idle"] = (open(synthetic_audio["tone"], "rb"), "tone.wav")
    r = post(client, data, auth)
    stages = r.get_json()["stages"]
    assert r.status_code == 422
    assert stages["audio_idle"]["code"] == "NOT_ENGINE"
    assert stages["audio_start"]["valid"] is True


def test_white_noise_is_rejected(client, auth, synthetic_audio):
    data = three_stages()
    data["audio_idle"] = (open(synthetic_audio["white_noise"], "rb"), "noise.wav")
    r = post(client, data, auth)
    assert r.status_code == 422
    assert r.get_json()["stages"]["audio_idle"]["code"] == "OTHER_SOUND"


def test_swapped_stages_give_warnings(client, auth):
    r = post(client, three_stages(start=HEALTHY_IDLE, idle=HEALTHY_START), auth)
    stages = r.get_json()["stages"]
    assert r.status_code == 200
    assert "warning" in stages["audio_start"]
    assert "warning" in stages["audio_idle"]


def test_legacy_single_file(client, auth):
    r = post(client, {"audio_file": audio(HEALTHY_IDLE)}, auth)
    assert r.status_code == 200
    assert r.get_json()["is_multi_stage"] is False


def test_deployment_test_endpoint(client, auth):
    r = client.get("/api/v1/test", headers=auth)
    assert r.status_code == 200
    assert r.get_json()["status"] == "success"


def test_uploads_are_deleted(client, auth, app_module):
    post(client, three_stages(), auth)
    assert os.listdir(app_module.UPLOAD_FOLDER) == []


# --- One guided recording covering start, idle and revving ---

def post_full(client, auth, recording):
    path, start_end, idle_end = recording
    data = {"audio_full": (open(path, "rb"), path.name)}
    if start_end is not None:
        data["start_end"], data["idle_end"] = str(start_end), str(idle_end)
    return post(client, data, auth)


def test_full_recording_healthy(client, auth, full_recordings):
    r = post_full(client, auth, full_recordings["healthy"])
    body = r.get_json()
    assert r.status_code == 200
    assert body["mode"] == "single_recording"
    assert body["fault_class"] == "healthy"
    assert set(body["stages"]) == {"audio_start", "audio_idle", "audio_acceleration"}
    assert body["stages"]["audio_start"]["detected"] == "start"


def test_full_recording_uses_the_app_stage_times(client, auth, full_recordings):
    _, start_end, idle_end = full_recordings["healthy"]
    stages = post_full(client, auth, full_recordings["healthy"]).get_json()["stages"]
    assert stages["audio_start"]["end"] == round(start_end, 2)
    assert stages["audio_idle"]["start"] == round(start_end, 2)
    assert stages["audio_acceleration"]["start"] == round(idle_end, 2)


def test_full_recording_healthy_with_late_start(client, auth, full_recordings):
    # Used to come back "knocking": the idle stage mixed the start's tail with idle.
    body = post_full(client, auth, full_recordings["late_start"]).get_json()
    assert body["fault_class"] == "healthy"
    assert body["color_indicator"] == "green"


def test_full_recording_fault_in_one_stage(client, auth, full_recordings):
    body = post_full(client, auth, full_recordings["knocking"]).get_json()
    assert body["fault_class"] == "knocking"
    assert body["color_indicator"] == "red"


def test_full_recording_without_engine_start_warns(client, auth, full_recordings):
    r = post_full(client, auth, full_recordings["no_start"])
    assert r.status_code == 200
    assert "warning" in r.get_json()["stages"]["audio_start"]


def test_full_recording_too_short(client, auth, full_recordings):
    r = post_full(client, auth, full_recordings["too_short"])
    assert r.status_code == 422
    assert r.get_json()["stages"]["audio_full"]["code"] == "TOO_SHORT"


def test_full_recording_of_noise_is_rejected(client, auth, full_recordings):
    r = post_full(client, auth, full_recordings["white_noise"])
    assert r.status_code == 422
    assert r.get_json()["stages"]["audio_full"]["code"] == "OTHER_SOUND"
