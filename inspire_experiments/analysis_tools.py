from __future__ import annotations

import time
import json
import copy
import matplotlib
from typing import List
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import itertools as it

from matplotlib.colors import to_rgba
# import matplotlib
# matplotlib.style.use('fivethirtyeight')
matplotlib.style.use('seaborn')

DEFAULT_DATA_DIR = './data/'
DEFAULT_PLOT_DIR = './plots/'


def get_expected_value(operator, state, n):
    m = 1
    for i in range(n):
        if operator[i] == 'Z' and state[i] == '1':
            m *= -1
    return m


def gen_M_matrix(n):
    # List of different Operators
    ops = ['I', 'Z']
    Operators = [''.join(op) for op in it.product(ops, repeat=n)]
    # List of calibration points
    states = ['0', '1']
    Cal_points = [''.join(s) for s in it.product(states, repeat=n)]
    # Calculate M matrix
    M = np.zeros((2 ** n, 2 ** n), dtype=int)
    for j, state in enumerate(Cal_points):
        Betas = np.ones(len(Operators))
        for i in range(2 ** n):
            Betas[i] = get_expected_value(Operators[i], state, n)
        M[j] = Betas
    M = np.linalg.pinv(M)  # invert matrix
    return M


def get_beta_matrix(cal_shots_dig, n):
    # List of different Operators
    ops = ['I', 'Z']
    Operators = [''.join(op) for op in it.product(ops, repeat=n)]
    # List of qubits
    Qubits = list(cal_shots_dig.keys())
    # Calculate Beta matrix
    H = {}
    B = {}
    M = gen_M_matrix(n)
    for op in Operators[1:]:
        H[op] = np.zeros(2 ** n)
        for i, state in enumerate(cal_shots_dig[Qubits[0]].keys()):
            correlator = 1
            for j, qubit in enumerate(Qubits):
                if op[j] == 'Z':
                    correlator *= np.array(cal_shots_dig[Qubits[j]][state])
            H[op][i] = np.mean(correlator)
        B[op] = np.dot(M, H[op])
    return B


def gen_gate_order(n):
    # Gate order in experiment
    tomo_gates = ['Z', 'X', 'Y']
    gate_order = [''.join(op)[::-1] for op in it.product(tomo_gates, repeat=n)]
    return np.array(gate_order)


def gen_n_Q_pauli(n):
    # Single qubit pauli terms
    Pauli_operators = {}
    Pauli_operators['I'] = np.array([[1, 0],
                                     [0, 1]])
    Pauli_operators['Z'] = np.array([[1, 0],
                                     [0, -1]])
    Pauli_operators['X'] = np.array([[0, 1],
                                     [1, 0]])
    Pauli_operators['Y'] = np.array([[0, -1j],
                                     [1j, 0]])
    # Four qubit pauli terms
    pauli_ops = ['I', 'X', 'Y', 'Z']
    Pauli_terms = {}
    Operators = [''.join(op) for op in it.product(pauli_ops, repeat=n)]
    for Op in Operators:
        Pauli_terms[Op] = Pauli_operators[Op[0]]
        for op in Op[1:]:
            Pauli_terms[Op] = np.kron(Pauli_terms[Op], Pauli_operators[op])
    return Pauli_terms


def get_pauli_expectation_values(
        tomo_shots_dig: dict,
        beta_matrix: dict,
        gate_order: np.ndarray | list,
        mask: np.ndarray = None
):
    '''
    Calculates Pauli expectation values (PEVs) in three steps:
        1. Calculate raw PEVs.
        2. Condition (post-select) data on no errors in stabilizers.
        3. Apply readout corrections to PEVs based on Beta matrix.
    '''
    Qubits = list(tomo_shots_dig.keys())  # [1:]
    n = len(Qubits)
    # print(Qubits, n)

    B_matrix = np.array([beta_matrix[key][1:] for key in beta_matrix.keys()])
    B_0 = np.array([beta_matrix[key][0] for key in beta_matrix.keys()])
    iB_matrix = np.linalg.inv(B_matrix)
    pauli_ops = ['I', 'X', 'Y', 'Z']
    P_values = {''.join(op): [] for op in it.product(pauli_ops, repeat=n)}
    P_frac = copy.deepcopy(P_values)
    for i, pre_rotation in enumerate(gate_order[:]):
        combs = [('I', op) for op in pre_rotation]
        P_vector = {''.join(o): 1 for o in it.product(*combs)}
        for correlator in P_vector.keys():
            # Calculate raw PEVs
            C = 1
            for j, qubit in enumerate(Qubits):
                if correlator[n - j - 1] != 'I':
                    # C *= np.array(tomo_shots_dig[qubit][i], dtype=float)
                    C *= np.array(tomo_shots_dig[qubit][pre_rotation], dtype=float)
            # Post-select data on stabilizer measurements
            if mask is not None:
                C *= mask[i]

            # fix to allow single qubit tomography (where assignment in above loop will not happen)
            if isinstance(C, int):
                C = np.array([C])

            n_total = len(C)
            C = C[~np.isnan(C)]
            n_selec = len(C)
            P_vector[correlator] = np.mean(C)
            P_frac[correlator] = n_selec / n_total
        # Apply readout corrections
        P = np.array([P_vector[key] for key in list(P_vector.keys())[1:]])
        P_corrected = np.dot(P - B_0, iB_matrix)
        P_vec_corr = {key: P_corrected[i - 1] if i != 0 else 1 for i, key in enumerate(list(P_vector.keys()))}
        # Fill main pauli vector with corresponding expectation values
        for key in P_vec_corr.keys():
            P_values[key].append(P_vec_corr[key])
    # Average all repeated pauli terms
    for key in P_values:
        P_values[key] = np.mean(P_values[key])
    # Calculate density matrix
    Pauli_terms_n = gen_n_Q_pauli(n)
    rho = np.zeros((2 ** n, 2 ** n)) * (1 + 0 * 1j)
    for op in Pauli_terms_n.keys():
        rho += P_values[op] * Pauli_terms_n[op] / 2 ** n
    return P_values, rho, P_frac


def fidelity(
        rho_1: np.ndarray,
        rho_2: np.ndarray,
        trace_conserved: bool = True
) -> float:
    if trace_conserved:
        if np.round(np.trace(rho_1), 3) != 1:
            raise ValueError('rho_1 unphysical, trace =/= 1, but ', np.trace(rho_1))
        if np.round(np.trace(rho_2), 3) != 1:
            raise ValueError('rho_2 unphysical, trace =/= 1, but ', np.trace(rho_2))
    sqrt_rho_1 = sp.linalg.sqrtm(rho_1)
    eig_vals = sp.linalg.eig(np.dot(np.dot(sqrt_rho_1, rho_2), sqrt_rho_1))[0]
    pos_eig = [vals for vals in eig_vals if vals > 0]
    return float(np.sum(np.real(np.sqrt(pos_eig)))) ** 2


def plot_density_matrix(
        rho: np.ndarray,
        rho_id: np.ndarray = None,
        rho2: np.ndarray = None,
        rho2_id: np.ndarray = None,
        title: str = '',
        title2: str = '',
        fidelity: float = None,
        fidelity2: float = None,
        angle: float = None,
        angle2: float = None,
        angle_text: str = '',
        angle_text2: str = '',
        ps_frac: float = None,
        ps_frac2: float = None,
        nr_shots: int = None,
        camera_azim: float = -55,
        camera_elev: float = 35,
        **kw,
):
    if rho2 is not None:
        figsize = (9, 8)
        fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=200, squeeze=False,# constrained_layout=True
                                subplot_kw=dict(projection='3d', azim=camera_azim, elev=camera_elev))
    else:
        figsize = (6, 5)
        fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=200, squeeze=False,# constrained_layout=True
                                subplot_kw=dict(projection='3d', azim=camera_azim, elev=camera_elev))

    for i, (rho, rho_id, title, angle, angle_text, fidelity, ps_frac) \
            in enumerate([(rho, rho_id, title, angle, angle_text, fidelity, ps_frac),
                        (rho2, rho2_id, title2, angle2, angle_text2, fidelity2, ps_frac2)]):
        if rho is None:
            continue

        ax = axs.flatten()[i]
        n = len(rho)
        labelsize = 6 if rho2 is not None else 9
        # xedges = np.arange(-.75, n, 1)
        # yedges = np.arange(-.75, n, 1)
        xedges = np.linspace(0, 1, n + 1)
        yedges = np.linspace(0, 1, n + 1)
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0
        dx = dy = 1 / n * 0.8
        dz = np.abs(rho).ravel()
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C3",'darkseagreen',"C0",'antiquewhite',"C3"])
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["firebrick",'forestgreen','snow',"royalblue","firebrick"])
        # cmap = matplotlib.cm.get_cmap('seismic')
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                                   ["firebrick", 'darkseagreen', 'snow', "steelblue",
                                                                    "firebrick"])
        norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        # norm = matplotlib.colors.CenteredNorm(vcenter=0, halfrange=np.pi)
        color = cmap(norm(np.angle(rho).ravel()))
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='max',
                 color=color, alpha=0.8, edgecolor='black', linewidth=.1)

        if rho_id is not None:
            color_id = cmap(norm(np.angle(rho_id).ravel()))
            dz1 = np.abs(rho_id).ravel()
            # selector
            s = [k for k in range(len(dz1)) if dz1[k] > .15]
            colors = [to_rgba(color_id[k], 0.3) if dz1[k] >= dz[k] else to_rgba(color_id[k], 1) for k in s]
            Z = [dz[k] if dz1[k] >= dz[k] else dz1[k] for k in s]
            DZ = [dz1[k] - dz[k] if dz1[k] >= dz[k] else dz[k] - dz1[k] for k in s]
            ax.bar3d(xpos[s], ypos[s], Z, dx, dy, dz=DZ, zsort='min',
                     color=colors, edgecolor=to_rgba('black', .25), linewidth=0.5)

        N = int(np.log2(n))
        states = ['0', '1']
        combs = [''.join(s) for s in it.product(states, repeat=N)]
        tick_period = n // (3 - 2 * (N % 2)) - N % 2
        ax.set_xticks(xpos[::n][::tick_period] + 1 / n / 2)
        ax.set_yticks(ypos[:n:tick_period] + 1 / n / 2)
        ax.set_xticklabels(combs[::tick_period], rotation=20, fontsize=labelsize, ha='right')
        ax.set_yticklabels(combs[::tick_period], rotation=10, fontsize=labelsize)
        ax.tick_params(axis='x', which='major', pad=-6, labelsize=labelsize)
        ax.tick_params(axis='y', which='major', pad=-6, labelsize=labelsize)
        ax.tick_params(axis='z', which='major', pad=-2 if rho2 is not None else 0, labelsize=labelsize)
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")

        if (rho > 0.7).any() or (rho_id > 0.6).any():
            ax.set_zticks(np.linspace(0, 1, 9))
            ax.set_zticklabels(['0', '', '0.25', '', '0.5', '', '0.75', '', '1.0'])
            ax.set_zlim(0, 1.0)
        else:
            ax.set_zticks(np.linspace(0, 0.5, 5))
            ax.set_zticklabels(['0', '', '0.25', '', '0.5'])
            ax.set_zlim(0, 0.5)

        ax.set_zlabel(r'$|\rho|$',
                      labelpad=-6 if rho2 is not None else -2,
                      size=10 if rho2 is not None else 12,
                      rotation=45)
        ax.set_title(title, size=7 if rho2 is not None else 9)
        # Text box
        s = r'$F_{|\psi\rangle}=' + fr'{fidelity * 100:.1f}\%$' if fidelity is not None else ''
        s += '\n' + angle_text if angle_text \
            else '\n' + r'$\mathrm{arg}(\rho_{0,-1})=' + fr'{angle:.1f}^\circ$' if angle is not None else ''
        s += '\n' + r'$P_\mathrm{ps}=' + fr'{ps_frac * 100:.1f}\%$' if ps_frac is not None else ''

        props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.4)
        if s:
            ax.text(0.6, 0.7, 0.66, s, size=8 if rho2 is not None else 10, bbox=props, va='bottom')
            # ax.text(0.5, 0.5, 0.5, s, size=5, bbox=props, va='bottom')
            # ax.text(1, 0.4, 1, s, size=5, bbox=props, va='bottom')
            # ax.text2D(0.5, 0.5, s, size=5, bbox=props, va='bottom')

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axs, orientation='vertical',
                      shrink=0.4 if rho2 is not None else 0.9,
                      pad=0.05 if rho2 is not None else 0.1)
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'], fontsize=labelsize+1)
    cb.set_label(r'arg$(\rho)$', fontsize=labelsize+2, labelpad=-10)
    # cb.ax.tick_params(labelsize=7)

    # fig.subplots_adjust(bottom=0.01, top=0.9, right=0.9)
    fig.set_tight_layout(tight=True)
    return fig


def plot_pauli_exp_values(
        pauli_terms: dict | pd.DataFrame,
        bloch_vector_in_legend: bool = False,
        figsize: tuple[float] = (7, 4),
        title: str = None
) -> tuple[plt.Figure, plt.Axes]:

    if isinstance(pauli_terms, pd.DataFrame):
        if bloch_vector_in_legend:
            expvals = pauli_terms[:-2]
            blochvals = pauli_terms[-2:]
        else:
            expvals = pauli_terms

        ax = expvals.plot.bar(width=0.8, figsize=figsize, alpha=0.8, use_index=True)#, colormap='Dark2')
        if bloch_vector_in_legend:
            ax.legend([fr"{name}: {series.index[0]} = {series.iloc[0]:.3f}, "
                       fr"{series.index[1]} = {series.iloc[1]:.2f}$^\circ$"
                       for name, series in blochvals.iteritems()], loc=0)
        else:
            ax.legend([fr"{name}" for name in expvals.columns], loc=0)
        ax.hlines(0, *ax.get_xlim(), 'k', linewidth=0.5)
        ax.set_xticklabels(expvals.index.to_list(), rotation=0)
        ax.set_ylim(-1, 1)
        # ax.grid(True, alpha=0.5, linewidth=1)
        ax.set(ylabel="Expectation value", xlabel="Operator", title=title)
        fig = ax.get_figure()
        fig.set_dpi(180)
        return fig, ax
    else:
        magnitude = np.sqrt((np.array(list(pauli_terms.values()))**2).sum())
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
        ax.bar(pauli_terms.keys(), pauli_terms.values(), alpha=0.8)
        ax.hlines(0, *ax.get_xlim(), 'k', linewidth=0.5)
        ax.set_ylim(-1, 1)
        ax.set(ylabel="Expectation value", xlabel="Operator", title=title)
        ax.legend([f"Bloch vector magnitude:\n{magnitude:.3f}"], loc=0)
        return fig, ax
