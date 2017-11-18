from unittest import mock

from pytocl.protocol import Serializer, Client, State
from pytocl.driver import Driver
from pytocl.car import State as CarState, Command


def test_init_encoding():
    d = Driver(False)
    s = Serializer()

    data = {'init': d.range_finder_angles}
    encoded = s.encode(data, prefix='SCR')

    assert encoded == b'SCR(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)'


def test_decode_server_message():
    buffer = b'(angle 0.008838)' \
             b'(curLapTime 4.052)' \
             b'(damage 0)' \
             b'(distFromStart 1015.56)' \
             b'(distRaced 42.6238)' \
             b'(fuel 93.9356)' \
             b'(gear 3)' \
             b'(lastLapTime 0)' \
             b'(opponents 123.4 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200' \
             b' 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200)' \
             b'(racePos 1)' \
             b'(rpm 4509.31)' \
             b'(speedX 81.5135)' \
             b'(speedY 0.40771)' \
             b'(speedZ -2.4422)' \
             b'(track 4.3701 4.52608 5.02757 6.07753 8.25773 11.1429 13.451 16.712 21.5022' \
             b' 30.2855 51.8667 185.376 69.9077 26.6353 12.6621 8.2019 6.5479 5.82979 5.63029)' \
             b'(trackPos 0.126012)' \
             b'(wheelSpinVel 67.9393 68.8267 71.4009 71.7363)' \
             b'(z 0.336726)' \
             b'(focus 26.0077 27.9798 30.2855 33.0162 36.3006)'

    d = Serializer().decode(buffer)

    assert len(d) == 19
    assert d['angle'] == '0.008838'
    assert d['wheelSpinVel'] == ['67.9393', '68.8267', '71.4009', '71.7363']

    c = CarState(d)

    assert c.angle == 0.5063800993366215
    assert c.current_lap_time == 4.052
    assert c.damage == 0
    assert c.distance_from_start == 1015.56
    assert c.distance_raced == 42.6238
    assert c.fuel == 93.9356
    assert c.gear == 3
    assert c.last_lap_time == 0.0
    assert c.opponents == (123.4, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                           200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                           200, 200, 200, 200, 200, 200, 200, 200)
    assert c.race_position == 1
    assert c.rpm == 4509.31
    assert c.speed_x == 22.64263888888889
    assert c.speed_y == 0.11325277777777779
    assert c.speed_z == -0.6783888888888889
    assert c.distances_from_edge == (4.3701, 4.52608, 5.02757, 6.07753, 8.25773, 11.1429, 13.451,
                                     16.712, 21.5022, 30.2855, 51.8667, 185.376, 69.9077, 26.6353,
                                     12.6621, 8.2019, 6.5479, 5.82979, 5.63029)
    assert c.distance_from_center == 0.126012
    assert c.wheel_velocities == (3892.635153073154, 3943.4794278130635, 4090.970223435639,
                                  4110.1872278843275)
    assert c.z == 0.336726
    assert c.focused_distances_from_edge == (26.0077, 27.9798, 30.2855, 33.0162, 36.3006)
    assert c.focused_distances_from_egde_valid

    # fake bad focus value:
    c.focused_distances_from_edge = (-1.0, -1.0, -1.0, -1.0, -1.0)
    assert not c.focused_distances_from_egde_valid


@mock.patch('pytocl.protocol.socket.socket')
def test_special_messages(mock_socket_ctor):
    mock_socket = mock.MagicMock()
    mock_socket_ctor.return_value = mock_socket
    mock_driver = mock.MagicMock()
    mock_driver.range_finder_angles = Driver(False).range_finder_angles
    client = Client(driver=mock_driver)
    assert client.state is State.STOPPED

    mock_socket.recvfrom = mock.MagicMock(side_effect=[(b'***identified***', None),
                                                       (b'***restart***', None),
                                                       (b'***shutdown***', None)])

    client.run()
    assert client.state is State.STOPPED

    # not supported on server side
    assert mock_driver.on_restart.call_count == 1
    assert mock_driver.on_shutdown.call_count == 1


def test_buffer_regression_1():
    buffer = b'(angle 0.00585968)(curLapTime -0.982)(damage 0)(distFromStart 972.935)' \
             b'(distRaced 0)(fuel 94)(gear 0)(lastLapTime 0)(opponents 200 200 200 200 ' \
             b'200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 ' \
             b'200 200 200 200 200 200 200 200 200 200 200 200 200 200)(racePos 1)(rpm 942.478)' \
             b'(speedX 0.0206057)(speedY 0.000264679)(speedZ -0.000624058)' \
             b'(track 5.00273 5.18736 5.79627 7.10971 9.87425 13.5626 16.5008 20.583 26.1558 ' \
             b'33.4167 42.2603 52.3137 63.2249 17.1145 10.1787 7.03222 5.75109 5.16562 4.99744)' \
             b'(trackPos -0.000529385)(wheelSpinVel 0 0 0 0)(z 0.355918)(focus -1 -1 -1 -1 -1)\x00'

    s = Serializer()
    d = s.decode(buffer)
    assert d['z'] == '0.355918'


def test_encode_command():
    c = Command()
    buffer = Serializer().encode(c.actuator_dict)
    assert b'(accel 0.0)' in buffer
    assert b'(clutch 0)' in buffer
