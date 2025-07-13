import os
import sys
import time
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import openseespy.postprocessing.Get_Rendering as opsplt

import timeit

DEBUG = False


def taper_maker(dt, cut_sec, ndata):
    time = np.arange(0, dt * ndata, dt)
    taper = np.ones(ndata)
    point_1 = int(cut_sec / dt)
    point_2 = ndata - point_1

    taper[:point_1] = 0.5 * (1 + np.cos(2.0 * np.pi * (time[:point_1] / (2 * cut_sec)) + np.pi))
    taper[point_2:] = 0.5 * (1 + np.cos(2.0 * np.pi * (time[point_2:] - time[point_2]) / (2 * cut_sec)))

    return taper


def load_elcentro(column="NS", dtype: Any = None):
    length = 2048
    el_centro = np.load("el_centro.npy")[:length, :]
    return el_centro[:, 0], el_centro[:, 1]


def load_randomwave(length, taper1):
    dt = 0.01
    time = np.arange(0, length * dt, dt)[:length]
    sigma = np.random.uniform(0.1, 10)
    wave = np.random.normal(0, sigma, size=time.shape)
    wave *= taper1
    return time, wave


class AnalyzingError(Exception):
    def __init__(self, error_code: int = 0, *args: object) -> None:
        super().__init__(*args)
        self.error_code = error_code


def setup_analyzer():
    ops.algorithm("Linear", False, False, True)
    # ops.algorithm("Newton")
    ops.test("NormDispIncr", 1e-16, 10)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGeneral")


def analyze_static():
    ops.wipeAnalysis()
    setup_analyzer()
    ops.integrator("LoadControl", 1)
    ops.analysis("Static")
    res = ops.analyze(1)
    if res < 0:
        raise AnalyzingError(res)
    # opsplt.plot_model("nodes", "elements")


def analyze_transient(time_series, excitation, factor=0.01):
    ops.wipeAnalysis()

    # initialize time
    ops.loadConst("-time", 0.0)

    ops.timeSeries("Path", 2, "-values", *excitation, "-time", *time_series)
    # relative response
    # pattern('UniformExcitation', patternTag, dir, '-accel', accelSeriesTag, '-fact', fact)
    ops.pattern("UniformExcitation", 2, 1, "-accel", 2, "-fact", factor)

    setup_analyzer()

    h = 0.02
    eigen = ops.eigen(1)
    betaKcomm = 2 * h / np.sqrt(eigen[0])

    ops.rayleigh(0.0, 0.0, 0.0, betaKcomm)
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    accel = [ops.nodeAccel(4, 1)]
    strain = [
        [ops.eleResponse(2, "force")[2]],
        [ops.eleResponse(3, "force")[2]],
        [ops.eleResponse(4, "force")[5]],
        [ops.eleResponse(5, "force")[5]],
    ]

    if DEBUG:
        for i in range(1, 8):
            print(f"element {i} force: " + str(ops.eleResponse(i, "force")), file=sys.stderr)

    for t in np.diff(time_series):
        res = ops.analyze(1, t)
        if res < 0:
            raise AnalyzingError(res)

        accel.append(ops.nodeAccel(4, 1))
        strain[0].append(ops.eleResponse(2, "force")[2])
        strain[1].append(ops.eleResponse(3, "force")[2])
        strain[2].append(ops.eleResponse(4, "force")[5])
        strain[3].append(ops.eleResponse(5, "force")[5])

    return np.array(accel), np.array(strain) / (
            np.array([173e-6, 173e-6, 399e-6, 399e-6])[:, None] * E
    )


def setup_model(params: Optional[Dict[str, Any]] = None, **kwargs):
    if isinstance(params, dict):
        kwargs.update(params)
    params = kwargs

    import openseespy.opensees as ops
    ops.wipe()

    # dimensions and degrees of freedom
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    height = 4.0
    width = 6.350

    # nodes and coordinates
    # node(nodeTag, x_crd, y_crd)
    ops.node(1, 0.0, 0.0)
    ops.node(2, width, 0.0)
    ops.node(3, 0.0, height)
    ops.node(4, width, height)

    h_lower = 1.5
    ops.node(5, 0.0, h_lower)
    ops.node(6, width, h_lower)

    ops.node(9, 0.0, 0.0)
    ops.node(10, width, 0.0)

    ops.fix(9, 1, 1, 1)
    ops.fix(10, 1, 1, 1)

    beam_area = 46.78e-4
    lower_column_area = 36.0e-4
    upper_column_area = 66.67e-4
    # beam_stiffness = params.get("beam_stiffness", constants.E)

    # beam_stiffness = params.get("beam_stiffness", constants.E)
    beam_flexural_stiffness = params.get("beam_flexural_stiffness")

    # section(secType, secTag, E, A, Iz)
    # lower part of column, l=1500
    ops.section("Elastic", 1, E, lower_column_area, 1352.1e-8)  # 150x150, t=12, Z=173
    # upper part of column, l=2600
    ops.section("Elastic", 2, E, upper_column_area, 3990.1e-8)  # 200x200, t=9, Z=399
    # beam
    ops.section("Elastic", 3, E, beam_area, beam_flexural_stiffness / E)  # 300x150, H=300, B=150, t1=6.5, t2=9, Z=481

    ops.geomTransf("Linear", 1)

    # elements
    # element('elasticBeamColumn', eleTag, node1, node2, secTag, transfTag)
    # beam
    ops.element("elasticBeamColumn", 1, 3, 4, 3, 1)
    # column
    ops.element("elasticBeamColumn", 2, 1, 5, 1, 1)
    ops.element("elasticBeamColumn", 3, 2, 6, 1, 1)
    ops.element("elasticBeamColumn", 4, 5, 3, 2, 1)
    ops.element("elasticBeamColumn", 5, 6, 4, 2, 1)

    flexural_rigidity_left = params.get("flexural_rigidity_left")
    flexural_rigidity_right = params.get("flexural_rigidity_right")
    # uniaxialMaterial('Elastic', matTag, E)
    ops.uniaxialMaterial("Elastic", 1, flexural_rigidity_left)
    ops.uniaxialMaterial("Elastic", 2, flexural_rigidity_right)

    # element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs, '-doRayleigh',rFlag)

    ops.element("zeroLength", 6, 9, 1, "-mat", 1, "-dir", 6, "-doRayleigh", 1)
    ops.element("zeroLength", 7, 10, 2, "-mat", 2, "-dir", 6, "-doRayleigh", 1)
    ops.equalDOF(9, 1, 1, 2)
    ops.equalDOF(10, 2, 1, 2)
    load = 7000  # kg  at roof top

    half_mass = ((lower_column_area * h_lower * 2 + upper_column_area * (
            height - h_lower) * 2 + beam_area * width) * steel_density + load) / 2

    ops.mass(3, half_mass, zero, zero)
    ops.mass(4, half_mass, zero, zero)

    ops.timeSeries("Constant", 1)

    # pattern('Plain', patternTag, tsTag)
    ops.pattern("Plain", 1, 1)

    ##????????????????????????????????????????????????????????????????????????????????????????
    # eleLoad('-ele', eleTag, '-type', '-beamUniform', Wy)
    ops.eleLoad("-ele", 1, "-type", "-beamUniform", -steel_density * beam_area * g)
    ops.eleLoad("-ele", 2, "-type", "-beamUniform", -load * g / width)
    ##????????????????????????????????????????????????????????????????????????????????????????


def multi_sim(theta, taper, time_series, excitation, kmin=4, nstep=64, add_noise=False):
    if len(theta.shape) == 2:
        output = np.zeros((theta.shape[0], 5, 1, nstep))
        for i in range(theta.shape[0]):
            tf, _ = Simulator(theta[i, :], taper, time_series, excitation, kmin=kmin, nstep=nstep, add_noise=add_noise)
            output[i, :, 0, :] = np.array(tf)
    else:
        tf, _ = Simulator(theta, taper, time_series, excitation, kmin=kmin, nstep=nstep, add_noise=add_noise)
        output = np.zeros((1, 5, 1, nstep))
        output[0, :, 0, :] = tf
    return output


def Simulator(theta, taper, time_series, excitation, kmin=4, nstep=64, add_noise=False):
    l, r, b = theta
    sn_ratio = 40
    # bs = (10 ** b) * E * I
    bs = 10 ** b * E * I
    fl = 10 ** (l + 3)
    fr = 10 ** (r + 3)
    setup_model(
        {
            "beam_flexural_stiffness": bs,
            "flexural_rigidity_left": fl,
            "flexural_rigidity_right": fr,
        }
    )
    analyze_static()
    observed_excitation = np.copy(excitation)
    a, s = analyze_transient(time_series, excitation)
    s = s - np.mean(s, axis=-1, keepdims=True)
    a = a * 100 + observed_excitation
    if add_noise:
        observed_excitation += generate_noise(observed_excitation, sn_ratio)
        a += generate_noise(a, sn_ratio, axis=-1)
        for i in range(4):
            s[i, :] += generate_noise(s[i, :], sn_ratio, axis=-1)

    input_spec = np.fft.fft(observed_excitation * taper)
    accel_spec = np.fft.fft(a * taper)
    tf = []
    tf1 = np.abs(accel_spec[kmin:kmin + nstep] / input_spec[kmin:kmin + nstep])
    tf.append(tf1.tolist())
    for i in range(4):
        ft1 = np.fft.fft(s[i, :] * taper * 10 ** 6)
        tf1 = np.abs(ft1[kmin:kmin + nstep] / input_spec[kmin:kmin + nstep])
        tf.append(tf1.tolist())
    return tf, observed_excitation


def test(l, r, taper, time_series, excitation, kmin=4, nstep=64, add_noise=False):
    sn_ratio = 50
    # bs = (10 ** b) * E * I
    bs = 2 * E * I
    fl = 10 ** l
    fr = 10 ** r
    setup_model(
        {
            "beam_flexural_stiffness": bs,
            "flexural_rigidity_left": fl,
            "flexural_rigidity_right": fr,
        }
    )
    analyze_static()
    observed_excitation = np.copy(excitation)
    a, s = analyze_transient(time_series, excitation)
    s = s - np.mean(s, axis=-1, keepdims=True)
    a = a * 100 + observed_excitation
    if add_noise:
        observed_excitation += generate_noise(observed_excitation, sn_ratio)
        a += generate_noise(a, sn_ratio, axis=-1)
        for i in range(4):
            s[i, :] += generate_noise(s[i, :], sn_ratio, axis=-1)

    input_spec = np.fft.fft(observed_excitation * taper)
    accel_spec = np.fft.fft(a * taper)
    tf = []
    tf1 = np.abs(accel_spec[kmin:kmin + nstep] / input_spec[kmin:kmin + nstep])
    tf.append(tf1.tolist())
    for i in range(4):
        ft1 = np.fft.fft(s[i, :] * taper * 10 ** 6)
        tf1 = np.abs(ft1[kmin:kmin + nstep] / input_spec[kmin:kmin + nstep])
        tf.append(tf1.tolist())
    return tf, observed_excitation


rng = np.random.default_rng()


def generate_noise(array, sn_ratio=0, axis=None):
    noise = rng.normal(
        0.0, np.sqrt(np.var(array, axis=axis, keepdims=True) * 10 ** (-sn_ratio / 10)), array.shape
    )
    return noise


##constants
g = 9.8  # m/s2
E = 205.0e9  # Pa
steel_density = 7850  # kg/m^3
I = 7210.0e-8  # m^4
zero = 1.0e-20  # almost zero
