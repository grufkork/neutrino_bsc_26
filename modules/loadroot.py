import uproot
import numpy as np

def load_events_file(path):
    f = uproot.open(path)
    
    e_ins = f['treeout;1']['e/in'].array()
    e_outs = f['treeout;1']['e/out'].array()
    
    q_vals = []
    omega_vals = []
    for (e_in,e_out) in zip(e_ins, e_outs):
        # for particle_idx in range(len(e["out.t"])):
        p_in = e_in
        p_out = e_out
        m_in = np.array([p_in["in.t"][0], p_in["in.x"][0], p_in["in.y"][0], p_in["in.z"][0]])
        m_out = np.array([p_out["out.t"][0], p_out["out.x"][0], p_out["out.y"][0], p_out["out.z"][0]])

        m_diff = m_in-m_out

        omega = m_diff[0]
        mom_diff_norm = np.linalg.norm(m_diff[1:4])
        # q = mom4_square(m_in-m_out)
        q_vals.append(mom_diff_norm)
        omega_vals.append(omega)

    return q_vals, omega_vals

def mom4_square(mom4):
    return np.sqrt(mom4[0]**2 - mom4[1]**2 - mom4[2]**2 - mom4[3]**2)