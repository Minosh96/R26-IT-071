"""
Builds every diagram used in the thesis in two forms from one description:
    thesis/diagrams/<name>.drawio   editable source (open with draw.io / diagrams.net)
    thesis/diagrams/png/<name>.png  the image embedded in the Word document

Run from the component root:
    venv/Scripts/python thesis/analysis/build_diagrams.py
"""
import os
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path

OUT = "thesis/diagrams"
PNG = os.path.join(OUT, "png")
os.makedirs(PNG, exist_ok=True)

INK = "#1F3B57"
WHITE = "#FFFFFF"
BLUE = "#DCE6F0"
SAND = "#FBF1E3"
GREY = "#F4F6F8"
GSTROKE = "#8A9BB0"
FONT = "Times New Roman"


class Diagram:
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.nodes, self.edges, self.order = {}, [], []

    # ------------------------------------------------------------ building
    def node(self, nid, x, y, w, h, text, shape="rect", fill=WHITE, bold=False, fs=13,
             dashed=False, valign="middle", stroke=INK):
        self.nodes[nid] = dict(id=nid, x=x, y=y, w=w, h=h, text=text, shape=shape, fill=fill,
                               bold=bold, fs=fs, dashed=dashed, valign=valign, stroke=stroke)
        self.order.append(nid)

    def edge(self, a, b, sa="b", sb="t", label=None, pts=None, dashed=False, arrow=True,
             lpos=0.5, loff=(0, 0), fs=12):
        self.edges.append(dict(a=a, b=b, sa=sa, sb=sb, label=label, pts=pts, dashed=dashed,
                               arrow=arrow, lpos=lpos, loff=loff, fs=fs, free=False))

    def line(self, p1, p2, label=None, dashed=False, arrow=True, lpos=0.5, loff=(0, -9), fs=12):
        self.edges.append(dict(p1=p1, p2=p2, label=label, dashed=dashed, arrow=arrow,
                               lpos=lpos, loff=loff, fs=fs, free=True, pts=None))

    # ------------------------------------------------------------ geometry
    @staticmethod
    def _side(s):
        return (s, 0.5) if isinstance(s, str) else s

    def _anchor(self, nid, side):
        n = self.nodes[nid]
        s, f = self._side(side)
        if s == "t":
            return (n["x"] + n["w"] * f, n["y"])
        if s == "b":
            return (n["x"] + n["w"] * f, n["y"] + n["h"])
        if s == "l":
            return (n["x"], n["y"] + n["h"] * f)
        return (n["x"] + n["w"], n["y"] + n["h"] * f)

    def _route(self, e):
        if e["free"]:
            return [e["p1"], e["p2"]], []
        p1, p2 = self._anchor(e["a"], e["sa"]), self._anchor(e["b"], e["sb"])
        if e["pts"] is not None:
            mid = list(e["pts"])
        else:
            va = self._side(e["sa"])[0] in "tb"
            vb = self._side(e["sb"])[0] in "tb"
            if va and vb:
                mid = [] if abs(p1[0] - p2[0]) < 0.5 else [(p1[0], (p1[1] + p2[1]) / 2), (p2[0], (p1[1] + p2[1]) / 2)]
            elif not va and not vb:
                mid = [] if abs(p1[1] - p2[1]) < 0.5 else [((p1[0] + p2[0]) / 2, p1[1]), ((p1[0] + p2[0]) / 2, p2[1])]
            elif va:
                mid = [(p1[0], p2[1])]
            else:
                mid = [(p2[0], p1[1])]
        return [p1] + mid + [p2], mid

    @staticmethod
    def _point_at(path, frac):
        seg = [((path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2) ** 0.5
               for i in range(len(path) - 1)]
        target, run = sum(seg) * frac, 0
        for i, s in enumerate(seg):
            if run + s >= target and s > 0:
                t = (target - run) / s
                return (path[i][0] + (path[i + 1][0] - path[i][0]) * t,
                        path[i][1] + (path[i + 1][1] - path[i][1]) * t)
            run += s
        return path[-1]

    # ------------------------------------------------------------ draw.io
    def to_drawio(self):
        style = {
            "rect": "rounded=0;",
            "round": "rounded=1;arcSize=18;",
            "diamond": "rhombus;",
            "ellipse": "ellipse;",
            "cyl": "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=9;",
            "doc": "shape=document;boundedLbl=1;size=0.22;",
            "note": "shape=note;size=12;",
            "group": "rounded=0;",
            "text": "text;",
        }
        out = ['<mxfile host="app.diagrams.net" agent="thesis build script" version="24.7.5">',
               f'<diagram id="{self.name}" name="{self.name}">',
               f'<mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
               f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" '
               f'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>']
        for nid in self.order:
            n = self.nodes[nid]
            st = style[n["shape"]] + f"whiteSpace=wrap;html=1;fontFamily={FONT};fontSize={n['fs']};fontColor=#000000;"
            if n["shape"] == "text":
                st += "strokeColor=none;fillColor=none;"
            else:
                st += f"strokeColor={n['stroke']};fillColor={n['fill']};"
            if n["bold"]:
                st += "fontStyle=1;"
            if n["dashed"]:
                st += "dashed=1;"
            if n["valign"] == "top":
                st += "verticalAlign=top;spacingTop=2;"
            val = escape(n["text"]).replace("\n", "&lt;br&gt;").replace('"', "&quot;")
            out.append(f'<mxCell id="{nid}" value="{val}" style="{st}" vertex="1" parent="1">'
                       f'<mxGeometry x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" as="geometry"/></mxCell>')
        for i, e in enumerate(self.edges):
            path, mid = self._route(e)
            st = f"rounded=0;html=1;fontFamily={FONT};fontSize={e['fs']};fontColor=#000000;strokeColor={INK};labelBackgroundColor=#FFFFFF;"
            st += "endArrow=block;endFill=1;endSize=6;" if e["arrow"] else "endArrow=none;"
            if e["dashed"]:
                st += "dashed=1;"
            val = escape(e["label"] or "").replace("\n", "&lt;br&gt;").replace('"', "&quot;")
            geo = f'<mxGeometry x="{e["lpos"] * 2 - 1:.2f}" y="0" relative="1" as="geometry">'
            if e["free"]:
                geo += (f'<mxPoint x="{path[0][0]}" y="{path[0][1]}" as="sourcePoint"/>'
                        f'<mxPoint x="{path[1][0]}" y="{path[1][1]}" as="targetPoint"/>')
                ends = ""
            else:
                sa, fa = self._side(e["sa"])
                sb, fb = self._side(e["sb"])
                pos = {"t": lambda f: (f, 0), "b": lambda f: (f, 1), "l": lambda f: (0, f), "r": lambda f: (1, f)}
                ex, ey = pos[sa](fa)
                nx, ny = pos[sb](fb)
                st = "edgeStyle=orthogonalEdgeStyle;" + st + f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;entryX={nx};entryY={ny};entryDx=0;entryDy=0;"
                ends = f' source="{e["a"]}" target="{e["b"]}"'
                if mid:
                    geo += '<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in mid) + "</Array>"
            if e["loff"] != (0, 0):
                geo += f'<mxPoint x="{e["loff"][0]}" y="{e["loff"][1]}" as="offset"/>'
            geo += "</mxGeometry>"
            out.append(f'<mxCell id="e{i}" value="{val}" style="{st}" edge="1" parent="1"{ends}>{geo}</mxCell>')
        out.append("</root></mxGraphModel></diagram></mxfile>")
        with open(os.path.join(OUT, self.name + ".drawio"), "w", encoding="utf-8") as f:
            f.write("\n".join(out))

    # ------------------------------------------------------------ PNG
    def to_png(self):
        fig = plt.figure(figsize=(self.w / 100, self.h / 100))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")
        px = 0.75  # px to pt at 100 px per inch
        for nid in self.order:
            n = self.nodes[nid]
            x, y, w, h = n["x"], n["y"], n["w"], n["h"]
            kw = dict(facecolor=n["fill"], edgecolor=n["stroke"], linewidth=1.0,
                      linestyle="--" if n["dashed"] else "-", zorder=2)
            s = n["shape"]
            if s in ("rect", "group"):
                ax.add_patch(Rectangle((x, y), w, h, **kw))
            elif s == "round":
                r = min(w, h) * 0.18
                ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                                            boxstyle=f"round,pad={r},rounding_size={r}", **kw))
            elif s == "diamond":
                ax.add_patch(Polygon([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)], **kw))
            elif s == "ellipse":
                ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, **kw))
            elif s == "cyl":
                r = 9
                ax.add_patch(Ellipse((x + w / 2, y + h - r), w, 2 * r, **kw))
                ax.add_patch(Rectangle((x, y + r), w, h - 2 * r, facecolor=n["fill"], edgecolor="none", zorder=2))
                ax.plot([x, x], [y + r, y + h - r], color=n["stroke"], lw=1.0, zorder=3)
                ax.plot([x + w, x + w], [y + r, y + h - r], color=n["stroke"], lw=1.0, zorder=3)
                ax.add_patch(Ellipse((x + w / 2, y + r), w, 2 * r, **{**kw, "zorder": 3}))
            elif s == "doc":
                d = h * 0.22
                verts = [(x, y), (x + w, y), (x + w, y + h - d),
                         (x + w * 0.75, y + h - 2 * d), (x + w * 0.5, y + h - d * 0.2), (x + w * 0.25, y + h),
                         (x + w * 0.1, y + h + d * 0.2), (x, y + h - d * 0.4), (x, y + h - d), (x, y)]
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                         Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
                ax.add_patch(PathPatch(Path(verts, codes), **kw))
            elif s == "note":
                c = 12
                ax.add_patch(Polygon([(x, y), (x + w - c, y), (x + w, y + c), (x + w, y + h), (x, y + h)], **kw))
                ax.plot([x + w - c, x + w - c, x + w], [y, y + c, y + c], color=n["stroke"], lw=1.0, zorder=3)
            ty = y + h / 2
            va = "center"
            if n["valign"] == "top":
                ty, va = y + 6, "top"
            elif s == "cyl":
                ty = y + h / 2 + 5
            elif s == "doc":
                ty = y + h * 0.42
            ax.text(x + w / 2, ty, n["text"], ha="center", va=va, fontsize=n["fs"] * px, family=FONT,
                    fontweight="bold" if n["bold"] else "normal", linespacing=1.15, zorder=5)
        for e in self.edges:
            path, _ = self._route(e)
            xs, ys = [p[0] for p in path], [p[1] for p in path]
            ls = (0, (4, 3)) if e["dashed"] else "-"
            ax.plot(xs[:-1] if False else xs, ys, color=INK, lw=1.0, linestyle=ls, zorder=1, solid_capstyle="butt")
            if e["arrow"]:
                (x0, y0), (x1, y1) = path[-2], path[-1]
                ax.annotate("", xy=(x1, y1), xytext=(x1 - (x1 - x0) * 1e-3 - (0.01 if x1 == x0 and y1 == y0 else 0), y1 - (y1 - y0) * 1e-3),
                            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.22", color=INK, lw=0, shrinkA=0, shrinkB=0),
                            zorder=4)
                # arrow head drawn as an explicit triangle so that it is identical at any scale
                import math
                ang = math.atan2(y1 - y0, x1 - x0)
                L, Wd = 8, 3.6
                bx, by = x1 - L * math.cos(ang), y1 - L * math.sin(ang)
                ax.add_patch(Polygon([(x1, y1), (bx - Wd * math.sin(ang), by + Wd * math.cos(ang)),
                                      (bx + Wd * math.sin(ang), by - Wd * math.cos(ang))],
                                     facecolor=INK, edgecolor=INK, linewidth=0.5, zorder=4))
            if e["label"]:
                lx, ly = self._point_at(path, e["lpos"])
                ax.text(lx + e["loff"][0], ly + e["loff"][1], e["label"], ha="center", va="center",
                        fontsize=e["fs"] * px, family=FONT, zorder=6, linespacing=1.1,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        fig.savefig(os.path.join(PNG, self.name + ".png"), dpi=300)
        plt.close(fig)

    def build(self):
        self.to_drawio()
        self.to_png()
        print("built", self.name)


# =====================================================================
# 1. Overall system architecture
# =====================================================================
d = Diagram("01_system_architecture", 660, 560)
d.node("g_app", 10, 12, 640, 96, "Watinakama.LK mobile application (Flutter)", "group", GREY, bold=True, valign="top", stroke=GSTROKE, dashed=True)
for i, (nid, t) in enumerate([("a_form", "Vehicle\ndetails form"), ("a_vin", "VIN plate\ncapture"), ("a_body", "Body image\ncapture (5 views)"),
                              ("a_audio", "Engine audio\nrecording"), ("a_res", "Results\nscreen")]):
    d.node(nid, 54 + i * 118, 44, 108, 50, t, "round")
d.node("c1", 142, 180, 150, 70, "Component 1\nVIN authentication\nFastAPI, port 8000")
d.node("c2", 302, 180, 150, 70, "Component 2\nBody condition analysis\nFastAPI, port 8080")
d.node("c3", 462, 180, 150, 70, "Component 3\nEngine sound diagnosis\nFlask, port 5003")
d.node("c4", 150, 340, 380, 84, "Component 4: Market value analysis\nFlask, port 5004\nstacked regression pipeline + valuation rules", fill=BLUE, bold=True)
d.node("s1", 170, 480, 160, 62, "Trained pipeline\n(joblib)", "cyl")
d.node("s2", 350, 480, 160, 62, "Repair cost table\n(JSON)", "cyl")
d.edge("a_vin", "c1", pts=[(226, 137), (217, 137)], label="image", lpos=0.75, loff=(-24, 0))
d.edge("a_body", "c2", pts=[(344, 137), (377, 137)], label="images", lpos=0.75, loff=(27, 0))
d.edge("a_audio", "c3", pts=[(462, 137), (537, 137)], label="audio clips", lpos=0.8, loff=(38, 0))
d.edge("c1", "c4", sb=("t", 0.18), label="vin_status", lpos=0.5)
d.edge("c2", "c4", sb=("t", 0.5), label="body_score", lpos=0.3)
d.edge("c3", "c4", sb=("t", 0.82), label="fault_class,\nconfidence", lpos=0.5)
d.edge("a_form", "c4", sa="b", sb="l", pts=[(108, 382)], label="vehicle details,\nlisted price", lpos=0.66)
d.edge("c4", "a_res", sa="r", sb="r", pts=[(640, 382), (640, 69)], label="verdict, fair value,\noffer range", lpos=0.42, loff=(-50, 30))
d.edge("s1", "c4", sa="t", sb=("b", 0.263))
d.edge("s2", "c4", sa="t", sb=("b", 0.737))
d.node("t1", 258, 440, 164, 22, "loaded once at start-up", "text", fs=12)
d.build()

# =====================================================================
# 2. Component architecture: offline training and online serving
# =====================================================================
d = Diagram("02_component_architecture", 660, 690)
d.node("gl", 28, 12, 290, 584, "Offline: data preparation and training", "group", GREY, bold=True, valign="top", stroke=GSTROKE, dashed=True)
d.node("gr", 342, 12, 290, 584, "Online: valuation service", "group", GREY, bold=True, valign="top", stroke=GSTROKE, dashed=True)
left = [("l1", "ALTO data_Updated.xlsx\n343 advertised listings", "doc"),
        ("l2", "preprocessing/clean_data.py\nfilter, impute, encode, derive", "rect"),
        ("l3", "alto_clean.csv\n338 records, 10 columns", "doc"),
        ("l4", "preprocessing/augment_holistic.py\nadd condition features, apply deductions", "rect"),
        ("l5", "holistic_augmented.csv\n2,000 records, 14 columns", "doc"),
        ("l6", "models/advanced_model_optimization.py\nrandomised search, stacking, evaluation", "rect"),
        ("l7", "advanced_best_model.joblib\nfitted pipeline (29 MB)", "cyl")]
for i, (nid, t, s) in enumerate(left):
    d.node(nid, 48, 44 + i * 78, 250, 54 if s != "cyl" else 60, t, s, fs=12)
right = [("r1", "Mobile app or API gateway", "round", WHITE),
         ("r2", "Flask application (api/app.py)\nGET /api/v1/health, POST /api/v1/valuate", "rect", WHITE),
         ("r3", "Bearer-token check and\nrequired-field validation", "rect", WHITE),
         ("r4", "valuate(): VIN gate\naltered VIN stops the valuation", "rect", BLUE),
         ("r5", "Price prediction\npipeline.predict on 14 features", "rect", BLUE),
         ("r6", "Repair lookup, negotiation range,\nverdict and explanation", "rect", BLUE),
         ("r7", "JSON response", "round", WHITE)]
for i, (nid, t, s, f) in enumerate(right):
    d.node(nid, 362, 44 + i * 78, 250, 54, t, s, fill=f, fs=12)
for a, b in zip([x[0] for x in left][:-1], [x[0] for x in left][1:]):
    d.edge(a, b)
for a, b in zip([x[0] for x in right][:-1], [x[0] for x in right][1:]):
    d.edge(a, b)
d.node("rc", 250, 622, 160, 58, "data/repair_costs.json\nfield survey, 2026", "cyl", fs=12)
d.edge("l7", "r5", sa="r", sb="l", pts=[(330, 542), (330, 383)], label="loaded at\nstart-up", lpos=0.45, fs=11)
d.edge("rc", "l4", sa="l", sb="l", pts=[(14, 651), (14, 305)])
d.edge("rc", "r6", sa="r", sb="r", pts=[(646, 651), (646, 461)])
d.build()

# =====================================================================
# 3. Research process
# =====================================================================
d = Diagram("03_research_process", 640, 330)
steps = [("p1", 30, 30, "1. Problem study\nmarket review,\nliterature survey"),
         ("p2", 235, 30, "2. Data collection\n343 Alto listings,\nrepair-cost field survey"),
         ("p3", 440, 30, "3. Data preparation\ncleaning, encoding,\naugmentation"),
         ("p4", 440, 210, "4. Modelling\nbaselines, tuning,\nstacked ensemble"),
         ("p5", 235, 210, "5. Evaluation\nhold-out, grouped CV,\nsensitivity analysis"),
         ("p6", 30, 210, "6. Deployment\nFlask API, mobile\nintegration, testing")]
for nid, x, y, t in steps:
    d.node(nid, x, y, 170, 84, t, "round", fill=BLUE if nid in ("p4", "p5") else WHITE)
d.edge("p1", "p2", "r", "l")
d.edge("p2", "p3", "r", "l")
d.edge("p3", "p4", "b", "t")
d.edge("p4", "p5", "l", "r")
d.edge("p5", "p6", "l", "r")
d.edge("p5", "p3", sa="t", sb=("b", 0.2), pts=[(320, 162), (474, 162)], dashed=True, label="findings feed back", lpos=0.45)
d.build()

# =====================================================================
# 4. Data preparation flowchart
# =====================================================================
d = Diagram("04_data_preparation_flow", 650, 925)
cx, bw = 120, 270
d.node("s", 205, 10, 100, 30, "Start", "ellipse", fill=BLUE)
d.node("n1", cx, 62, bw, 44, "Load ALTO data_Updated.xlsx\n(343 rows, 11 columns)")
d.node("d1", cx + 5, 128, bw - 10, 66, "Listed price above\nLKR 10 million?", "diamond", fill=SAND)
d.node("x1", 455, 139, 175, 44, "Drop the row\n(2 rows removed)")
d.node("d2", cx + 5, 216, bw - 10, 66, "Year of manufacture\nwithin 2012–2017?", "diamond", fill=SAND)
d.node("x2", 455, 227, 175, 44, "Drop the row\n(3 rows removed)")
d.node("n2", cx, 304, bw, 56, "Encode owner count, BN/R and power\noptions; fill missing mileage with the\nmedian of the same model year")
d.node("n3", cx, 382, bw, 44, "Derive vehicle_age and reg_gap;\nrename and select ten columns")
d.node("f1", cx + 35, 448, bw - 70, 50, "alto_clean.csv\n(338 rows)", "doc", fill=BLUE)
d.node("n4", cx, 524, bw, 56, "Copy the 338 real rows unchanged and\nlabel them healthy, confidence 1.0,\nbody score 100")
d.node("n5", cx, 602, bw, 56, "Draw 1,662 rows with replacement;\nperturb mileage (s.d. 5%) and\nprice (s.d. 1.5%)")
d.node("n6", cx, 680, bw, 44, "Draw fault class, confidence and\nbody score for each drawn row")
d.node("n7", cx, 746, bw, 44, "Subtract engine and body deductions\nfrom price (floor LKR 0.5 million)")
d.node("f2", cx + 35, 812, bw - 70, 50, "holistic_augmented.csv\n(2,000 rows)", "doc", fill=BLUE)
d.node("e", 205, 884, 100, 30, "End", "ellipse", fill=BLUE)
for a, b in [("s", "n1"), ("n1", "d1"), ("n2", "n3"), ("n3", "f1"), ("f1", "n4"), ("n4", "n5"), ("n5", "n6"),
             ("n6", "n7"), ("n7", "f2"), ("f2", "e")]:
    d.edge(a, b)
d.edge("d1", "x1", "r", "l", label="Yes")
d.edge("d1", "d2", label="No", loff=(16, 0))
d.edge("d2", "x2", "r", "l", label="No")
d.edge("d2", "n2", label="Yes", loff=(18, 0))
d.build()

# =====================================================================
# 5. Stacking ensemble
# =====================================================================
d = Diagram("05_stacking_ensemble", 650, 520)
d.node("i", 215, 12, 220, 44, "Input row: 14 features\n(8 vehicle, 3 condition, 3 derived)", "round")
d.node("ct", 105, 84, 440, 50, "ColumnTransformer\nStandardScaler on 13 numeric columns, OneHotEncoder on fault_class", fill=GREY)
bl = [("b1", "Random Forest\n300 trees\nmax depth 10"), ("b2", "Gradient Boosting\n200 trees, rate 0.05\nmax depth 5"),
      ("b3", "XGBoost\n100 trees, rate 0.1\nmax depth 7"), ("b4", "LightGBM\n200 trees, rate 0.05\n50 leaves")]
for k, (nid, t) in enumerate(bl):
    d.node(nid, 22 + k * 156, 190, 138, 74, t, fill=BLUE)
    d.edge("ct", nid, sa=("b", 0.5), sb="t", pts=[(325, 162), (22 + k * 156 + 69, 162)])
    d.edge(nid, "oof", sa="b", sb="t", pts=[(22 + k * 156 + 69, 292), (325, 292)])
d.node("oof", 175, 320, 300, 44, "Out-of-fold predictions of the four learners\n(5-fold, training rows only)", fill=GREY)
d.node("meta", 215, 392, 220, 44, "Ridge regression meta-learner\n(alpha = 1.0)", fill=SAND)
d.node("o", 205, 464, 240, 44, "Predicted market value (LKR million)", "ellipse")
d.edge("i", "ct")
d.edge("oof", "meta")
d.edge("meta", "o")
d.build()

# =====================================================================
# 6. Valuation decision flow
# =====================================================================
d = Diagram("06_valuation_decision_flow", 660, 975)
cx, bw = 100, 280
d.node("s", 190, 8, 100, 30, "Start", "ellipse", fill=BLUE)
d.node("n1", cx, 58, bw, 58, "Receive vehicle details, listed price,\nvin_status, body_score, fault_class\nand confidence")
d.node("d1", cx + 10, 138, bw - 20, 66, "vin_status is\n\"altered\"?", "diamond", fill=SAND)
d.node("x1", 450, 146, 170, 50, "Return CHECK_VIN\nwith a VIN warning;\nno price shown")
d.node("d2", cx + 10, 228, bw - 20, 66, "vin_status needs review\nor is unknown?", "diamond", fill=SAND)
d.node("x2", 450, 239, 170, 44, "Attach a VIN caution\nnote to the response")
d.node("n2", cx, 320, bw, 46, "Build the 14-feature row (age, registration\ngap, log and per-year terms)")
d.node("n3", cx, 390, bw, 46, "Predict fair value with the stacked\npipeline (floor LKR 0.5 million)")
d.node("n4", cx, 460, bw, 46, "Look up engine and body repair ranges\nin the repair cost table")
d.node("n5", cx, 530, bw, 46, "Compute the negotiation range and\nthe recommended offer")
d.node("d3", cx + 10, 600, bw - 20, 66, "Listed price above\n1.05 × fair value?", "diamond", fill=SAND)
d.node("x3", 450, 611, 170, 44, "Verdict: OVERPRICED")
d.node("d4", cx + 10, 690, bw - 20, 66, "Listed price below\n0.95 × fair value?", "diamond", fill=SAND)
d.node("x4", 450, 701, 170, 44, "Verdict: GOOD_DEAL\n(offer set to listed price)")
d.node("n6", cx + 40, 782, bw - 80, 40, "Verdict: FAIR_PRICE")
d.node("n7", cx, 852, bw, 46, "Compose the explanation and\nreturn the JSON response")
d.node("e", 190, 928, 100, 30, "End", "ellipse", fill=BLUE)
for a, b in [("s", "n1"), ("n1", "d1"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "d3"), ("n6", "n7"), ("n7", "e")]:
    d.edge(a, b)
d.edge("d1", "x1", "r", "l", label="Yes")
d.edge("d1", "d2", label="No", loff=(16, 0))
d.edge("d2", "x2", "r", "l", label="Yes")
d.edge("d2", "n2", label="No", loff=(16, 0))
d.edge("x2", "n2", sa="b", sb="r", pts=[(535, 343)])
d.edge("d3", "x3", "r", "l", label="Yes")
d.edge("d3", "d4", label="No", loff=(16, 0))
d.edge("d4", "x4", "r", "l", label="Yes")
d.edge("d4", "n6", label="No", loff=(16, 0))
d.edge("x3", "n7", sa="r", sb=("r", 0.35), pts=[(642, 633), (642, 868.1)])
d.edge("x4", "n7", sa="b", sb=("r", 0.65), pts=[(535, 881.9)])
d.build()

# =====================================================================
# 7. Sequence diagram
# =====================================================================
d = Diagram("07_inspection_sequence", 700, 560)
parts = [("q_user", 50, "Buyer"), ("q_app", 165, "Mobile app"), ("q_c1", 285, "C1: VIN"),
         ("q_c2", 395, "C2: Body"), ("q_c3", 505, "C3: Engine"), ("q_c4", 625, "C4: Valuation")]
X = {}
for nid, x, t in parts:
    d.node(nid, x - 48, 12, 96, 34, t, fill=BLUE if nid == "q_c4" else WHITE, bold=(nid == "q_c4"))
    d.line((x, 46), (x, 545), dashed=True, arrow=False)
    X[nid] = x


def msg(a, b, y, text, ret=False, lpos=0.5):
    d.line((X[a], y), (X[b], y), label=text, dashed=ret, lpos=lpos, fs=11.5)


msg("q_user", "q_app", 84, "vehicle details, price")
msg("q_user", "q_app", 118, "photos and audio")
msg("q_app", "q_c1", 158, "POST /predict (image)")
msg("q_c1", "q_app", 188, "vin_status", ret=True)
msg("q_app", "q_c2", 228, "POST /analyze (5 images)")
msg("q_c2", "q_app", 258, "body_score", ret=True)
msg("q_app", "q_c3", 298, "POST /analyze (audio clips)")
msg("q_c3", "q_app", 328, "fault_class, confidence", ret=True)
msg("q_app", "q_c4", 372, "POST /api/v1/valuate (JSON body, bearer token)")
d.node("q_note", 548, 388, 146, 62, "VIN gate, prediction,\nrepair lookup, verdict", "note", fill=SAND, fs=11.5)
msg("q_c4", "q_app", 476, "verdict, fair value, negotiation range, explanation", ret=True)
msg("q_app", "q_user", 516, "results screen", ret=True)
d.build()

# =====================================================================
# 8. Conceptual framework
# =====================================================================
d = Diagram("08_conceptual_framework", 660, 396)
ins = ["Vehicle attributes\nyear, age, mileage, owners, origin, options", "VIN authenticity\n(Component 1)",
       "Body condition score, 0–100\n(Component 2)", "Engine fault class and confidence\n(Component 3)",
       "Local repair costs\n(field survey of sellers and garages)"]
for i, t in enumerate(ins):
    d.node(f"in{i}", 12, 18 + i * 74, 232, 54, t, fs=12)
d.node("core", 286, 118, 132, 160, "Condition-aware\nvaluation\n\nstacked tree\nensemble\n+\nrule layer", fill=BLUE, bold=True, fs=12.5)
outs = ["Fair market value (LKR)", "Verdict: good deal, fair price,\noverpriced or do not buy",
        "Negotiation range and\nrecommended offer", "Plain-language explanation"]
for i, t in enumerate(outs):
    d.node(f"out{i}", 460, 46 + i * 78, 188, 54, t, "round", fs=12)
for i in range(5):
    d.edge(f"in{i}", "core", sa="r", sb=("l", 0.1 + i * 0.2), pts=[(264, 45 + i * 74), (264, 118 + 160 * (0.1 + i * 0.2))])
for i in range(4):
    d.edge("core", f"out{i}", sa=("r", 0.14 + i * 0.24), sb="l", pts=[(438, 118 + 160 * (0.14 + i * 0.24)), (438, 73 + i * 78)])
d.build()

# =====================================================================
# 9. Deployment view
# =====================================================================
d = Diagram("09_deployment_view", 670, 380)
d.node("ph", 12, 118, 150, 150, "Smartphone\n(Android / iOS)", "group", GREY, bold=True, valign="top", stroke=GSTROKE, dashed=True)
d.node("app", 28, 176, 118, 66, "Watinakama.LK\nFlutter app\n(ApiService)", "round")
d.node("host", 236, 12, 422, 356, "Backend host (development PC on the same network, or a cloud VM)", "group", GREY, bold=True, valign="top", stroke=GSTROKE, dashed=True, fs=12)
d.node("h1", 252, 44, 122, 44, "VIN service\nport 8000", fs=12)
d.node("h2", 386, 44, 122, 44, "Body service\nport 8080", fs=12)
d.node("h3", 520, 44, 122, 44, "Engine service\nport 5003", fs=12)
d.node("g4", 252, 106, 390, 250, "Market valuation service, port 5004", "group", WHITE, bold=True, valign="top", fs=12)
d.node("k1", 268, 138, 358, 46, "Flask 3 application with flask-cors and flasgger (Swagger UI)", fill=BLUE, fs=12)
d.node("k2", 268, 204, 358, 40, "inference/valuate.py: valuation logic", fill=BLUE, fs=12)
d.node("k3", 268, 270, 112, 70, "advanced_best\n_model.joblib\n(29 MB)", "cyl", fs=11.5)
d.node("k4", 391, 270, 112, 70, "repair_costs\n.json", "cyl", fs=11.5)
d.node("k5", 514, 276, 112, 58, ".env\nAPI token, port", "note", fill=SAND, fs=11.5)
d.edge("app", "k1", sa="r", sb="l", pts=[(200, 209), (200, 161)], label="HTTP, JSON,\nbearer token", lpos=0.4, fs=11.5)
d.edge("k1", "k2")
d.edge("k3", "k2", sa="t", sb=("b", 0.156))
d.edge("k4", "k2", sa="t", sb=("b", 0.5))
d.edge("k5", "k1", sa="r", sb="r", pts=[(634, 305), (634, 161)], dashed=True)
d.build()

# =====================================================================
# 10. Verdict bands
# =====================================================================
d = Diagram("10_verdict_bands", 640, 150)
d.node("v1", 20, 40, 220, 50, "GOOD_DEAL\nlisted price below 0.95 × FV", fill=BLUE, fs=12.5)
d.node("v2", 240, 40, 160, 50, "FAIR_PRICE\nwithin ±5% of FV", fill=WHITE, fs=12.5)
d.node("v3", 400, 40, 220, 50, "OVERPRICED\nlisted price above 1.05 × FV", fill=SAND, fs=12.5)
d.node("v4", 200, 96, 80, 20, "0.95 × FV", "text", fs=12)
d.node("v5", 280, 96, 80, 20, "FV", "text", fs=12)
d.node("v6", 360, 96, 80, 20, "1.05 × FV", "text", fs=12)
d.node("v7", 150, 122, 340, 22, "Listed price relative to the predicted fair value (FV)", "text", fs=12)
d.node("v8", 150, 10, 340, 22, "An altered VIN returns CHECK_VIN before any price is compared", "text", fs=12)
d.line((320, 90), (320, 98), arrow=False)
d.build()
