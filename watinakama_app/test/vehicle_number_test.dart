import 'package:flutter_test/flutter_test.dart';
import 'package:watinakama_app/utils/vehicle_number.dart';

void main() {
  group('findVehicleNumber', () {
    test('finds a 17-character VIN', () {
      expect(findVehicleNumber('MA3EUA61S00617272'), 'MA3EUA61S00617272');
    });

    test('finds a VIN read with stray spaces', () {
      expect(findVehicleNumber('* MA3EUA61S 0061 7272 * CF'), 'MA3EUA61S00617272');
    });

    test('finds a Japanese-style chassis number', () {
      expect(findVehicleNumber('CHASSIS NO.\nHA36S-123456'), 'HA36S-123456');
    });

    test('finds an engine number', () {
      expect(findVehicleNumber('R06A - 1234567'), 'R06A-1234567');
    });

    test('ignores ordinary text', () {
      expect(findVehicleNumber('Happy birthday\nSee you at 7 pm\nCall 0771234567'), isNull);
    });

    test('ignores an empty photo', () {
      expect(findVehicleNumber(''), isNull);
    });
  });
}
