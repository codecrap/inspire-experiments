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
    ops = ['I','Z']
    Operators = [''.join(op) for op in it.product(ops, repeat=n)]
    # List of calibration points
    states = ['0','1']
    Cal_points = [''.join(s) for s in it.product(states, repeat=n)]
    # Calculate M matrix
    M = np.zeros((2**n, 2**n), dtype=int)
    for j, state in enumerate(Cal_points):
        Betas = np.ones(len(Operators))
        for i in range(2**n):
            Betas[i] = get_expected_value(Operators[i], state, n)
        M[j] = Betas
    M = np.linalg.pinv(M) # invert matrix
    return M

def get_Beta_matrix(Cal_shots_dig, n):
    # List of different Operators
    ops = ['I','Z']
    Operators = [''.join(op) for op in it.product(ops, repeat=n)]
    # List of qubits
    Qubits = list(Cal_shots_dig.keys())
    # Calculate Beta matrix
    H = {}
    B = {}
    M = gen_M_matrix(n)
    for op in Operators[1:]:
        H[op] = np.zeros(2**n)
        for i, state in enumerate(Cal_shots_dig[Qubits[0]].keys()):
            correlator = 1
            for j, qubit in enumerate(Qubits):
                if op[j] == 'Z':
                    correlator *= np.array(Cal_shots_dig[Qubits[j]][state])
            H[op][i] = np.mean(correlator)
        B[op] = np.dot(M, H[op])
    return B

def gen_gate_order(n):
    # Gate order in experiment
    tomo_gates = ['Z', 'X', 'Y']
    Gate_order = [''.join(op)[::-1] for op in it.product(tomo_gates, repeat=n)]
    return np.array(Gate_order)

def gen_n_Q_pauli(n):
    # Single qubit pauli terms
    Pauli_operators = {}
    Pauli_operators['I'] = np.array([[  1,  0],
                                     [  0,  1]])
    Pauli_operators['Z'] = np.array([[  1,  0],
                                     [  0, -1]])
    Pauli_operators['X'] = np.array([[  0,  1],
                                     [  1,  0]])
    Pauli_operators['Y'] = np.array([[  0,-1j],
                                     [ 1j,  0]])
    # Four qubit pauli terms
    pauli_ops = ['I', 'X', 'Y', 'Z']
    Pauli_terms = {}
    Operators = [''.join(op) for op in it.product(pauli_ops, repeat=n)]
    for Op in Operators:
        Pauli_terms[Op] = Pauli_operators[Op[0]]
        for op in Op[1:]:
            Pauli_terms[Op] = np.kron(Pauli_terms[Op], Pauli_operators[op])
    return Pauli_terms

def get_Pauli_expectation_values(Beta_matrix, Gate_order, Mask, Tomo_shots_dig):
    '''
    Calculates Pauli expectation values (PEVs) in three steps:
        1. Calculate raw PEVs.
        2. Condition (post-select) data on no errors in stabilizers.
        3. Apply readout corrections to PEVs based on Beta matrix.
    '''
    Qubits = list(Tomo_shots_dig.keys()) #[1:]
    n = len(Qubits)
    # print(Qubits, n)

    B_matrix = np.array([Beta_matrix[key][1:] for key in Beta_matrix.keys()])
    B_0 = np.array([Beta_matrix[key][0] for key in Beta_matrix.keys()])
    iB_matrix = np.linalg.inv(B_matrix)
    pauli_ops = ['I', 'X', 'Y', 'Z']
    P_values = {''.join(op):[] for op in it.product(pauli_ops, repeat=n)}
    P_frac = copy.deepcopy(P_values)
    for i, pre_rotation in enumerate(Gate_order[:]):
        combs = [('I', op) for op in pre_rotation ]
        P_vector = {''.join(o):1 for o in it.product(*combs)}
        for correlator in P_vector.keys():
            # Calculate raw PEVs
            C = 1
            for j, qubit in enumerate(Qubits):
                if correlator[n-j-1] != 'I':
                    # C *= np.array(Tomo_shots_dig[qubit][i], dtype=float)
                    C *= np.array(Tomo_shots_dig[qubit][pre_rotation], dtype=float)
            # Post-select data on stabilizer measurements
            C = C*Mask[i]
            n_total = len(C)
            C = C[~np.isnan(C)]
            n_selec = len(C)
            P_vector[correlator] = np.mean(C)
            P_frac[correlator] = n_selec/n_total
        # Aplly readout corrections
        P = np.array([P_vector[key] for key in list(P_vector.keys())[1:]])
        P_corrected = np.dot(P-B_0, iB_matrix)
        P_vec_corr = { key: P_corrected[i-1] if i!=0 else 1 for i, key in enumerate(list(P_vector.keys()))}
        # Fill main pauli vector with corresponding expectation values
        for key in P_vec_corr.keys():
            P_values[key].append(P_vec_corr[key])
    # Average all repeated pauli terms
    for key in P_values:
        P_values[key] = np.mean(P_values[key])
    # Calculate density matrix
    Pauli_terms_n = gen_n_Q_pauli(n)
    rho = np.zeros((2**n,2**n))*(1+0*1j)
    for op in Pauli_terms_n.keys():
        rho += P_values[op]*Pauli_terms_n[op]/2**n
    return P_values, rho, P_frac

def fidelity(rho_1, rho_2, trace_conserved = False):
    if trace_conserved:
        if np.round(np.trace(rho_1), 3) !=1:
            raise ValueError('rho_1 unphysical, trace =/= 1, but ', np.trace(rho_1))
        if np.round(np.trace(rho_2), 3) !=1:
            raise ValueError('rho_2 unphysical, trace =/= 1, but ', np.trace(rho_2))
    sqrt_rho_1 = sp.linalg.sqrtm(rho_1)
    eig_vals = sp.linalg.eig(np.dot(np.dot(sqrt_rho_1,rho_2),sqrt_rho_1))[0]
    pos_eig = [vals for vals in eig_vals if vals > 0]
    return float(np.sum(np.real(np.sqrt(pos_eig))))**2


def plot_density_matrix(
        rho,
        ax,
        rho_id=None,
        title='',
        fidelity=None,
        angle=None,
        angle_text='',
        ps_frac=None,
        nr_shots=None,
        **kw,
):
    fig = ax.get_figure()
    n = len(rho)
    # xedges = np.arange(-.75, n, 1)
    # yedges = np.arange(-.75, n, 1)
    xedges = np.linspace(0, 1, n+1)
    yedges = np.linspace(0, 1, n+1)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 1/n*0.8
    dz = np.abs(rho).ravel()
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C3",'darkseagreen',"C0",'antiquewhite',"C3"])
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["firebrick",'forestgreen','snow',"royalblue","firebrick"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["firebrick",'darkseagreen','snow',"steelblue","firebrick"])
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black",'blue',"white",'red',"black"])
    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    # cmap = matplotlib.cm.get_cmap('seismic')
    # norm = matplotlib.colors.CenteredNorm(vcenter=0, halfrange=np.pi)
    # color = cmap(norm([np.angle(e) for e in rho.ravel()]))
    color = cmap(norm(np.angle(rho).ravel()))
    # print(list(zip(np.angle(rho).ravel(), norm(np.angle(rho).ravel()))))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='max',
             color=color, alpha=0.8, edgecolor='black', linewidth=.1)

    if rho_id is not None:
        # color_id = cmap(norm([np.angle(e) for e in rho_id.ravel()]))
        color_id = cmap(norm(np.angle(rho_id).ravel()))
        dz1 = np.abs(rho_id).ravel()
        # selector
        s = [k for k in range(len(dz1)) if dz1[k] > .15]
        colors = [ to_rgba(color_id[k], 0.3) if dz1[k] >= dz[k] else to_rgba(color_id[k], 1) for k in s ]
        Z = [ dz[k] if dz1[k] >= dz[k] else dz1[k] for k in s ]
        DZ = [ dz1[k]-dz[k] if dz1[k] >= dz[k] else dz[k]-dz1[k] for k in s ]
        ax.bar3d(xpos[s], ypos[s], Z, dx, dy, dz=DZ, zsort='min',
                 color=colors, edgecolor=to_rgba('black', .25), linewidth=0.5)

    N = int(np.log2(n))
    states = ['0', '1']
    combs = [''.join(s) for s in it.product(states, repeat=N)]
    tick_period = n//(3-2*(N%2)) - N%2
    ax.set_xticks(xpos[::n][::tick_period]+1/n/2)
    ax.set_yticks(ypos[:n:tick_period]+1/n/2)
    ax.set_xticklabels(combs[::tick_period], rotation=20, fontsize=6, ha='right')
    ax.set_yticklabels(combs[::tick_period], rotation=-40, fontsize=6)
    ax.tick_params(axis='x', which='major', pad=-6, labelsize=6)
    ax.tick_params(axis='y', which='major', pad=-6, labelsize=6)
    ax.tick_params(axis='z', which='major', pad=-2, labelsize=6)
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

    ax.set_zlabel(r'$|\rho|$', labelpad=-8, size=7, rotation=45)
    ax.set_title(title, size=7)
    # Text box
    s = r'$F_{|\psi\rangle}='+fr'{fidelity*100:.1f}\%$'
    s += '\n' + angle_text if angle_text else '\n' + r'$\mathrm{arg}(\rho_{0,15})='+fr'{angle:.1f}^\circ$' if angle else ''
    s += '\n' + r'$P_\mathrm{ps}='+fr'{ps_frac*100:.1f}\%$' if ps_frac else ''

    # s = ''.join((r'$F_{|\psi\rangle}='+fr'{fidelity*100:.1f}\%$', '\n',
    #              angle_text))
                 # r'$\mathrm{arg}(\rho_{0,15})='+fr'{angle:.1f}^\circ$'))
                 # '\n',
                 # r'$P_\mathrm{ps}='+fr'{ps_frac*100:.1f}\%$', '\n',
                 # f'# shots per Pauli {nr_shots}'))
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5)
    ax.text(1, 0.4, 1, s, size=5, bbox=props, va='bottom')
    # ax.text(0, 0, 1, s, size=5, bbox=props, va='bottom')
    # colorbar
    # fig.subplots_adjust(bottom=0.1)
    # cbar_ax = fig.add_axes([0.55, 0.56, 0.01, 0.275])
    # cbar_ax = fig.add_axes([0.55, 0.3, 0.01, 0.4])
    cbar_ax = fig.add_axes([0.675, 0.369, 0.015, 0.25])
    # cbar_ax = fig.add_subplot(133)
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C3",'darkseagreen',"C0",'antiquewhite',"C3"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                          orientation='vertical')
    # cb = plt.colorbar(sm, ax=ax, orientation='vertical')
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
    cb.set_label(r'arg$(\rho)$', fontsize=7, labelpad=-10)
    cb.ax.tick_params(labelsize=7)
