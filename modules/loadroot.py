import uproot
import numpy as np

def load_events_file(path):
    f = uproot.open(path)

    keys = f.keys()
    # keys = ["treeout;1"]
    q_vals = []
    omega_vals = []
    
    mom_ins = []
    mom_outs = []
    
    weights = []

    for key in keys:
        # for k in f[key].keys():
        #     print(key + " " + k)
        if not key.startswith("treeout;"):
            continue
        e_ins = f[key]['e/in'].array()
        e_outs = f[key]['e/out'].array()
        for (e_in,e_out) in zip(e_ins, e_outs):
            # for particle_idx in range(len(e["out.t"])):
            p_in = e_in
            p_out = e_out
            m_in = np.array([p_in["in.t"][0], p_in["in.x"][0], p_in["in.y"][0], p_in["in.z"][0]])
            m_out = np.array([p_out["out.t"][0], p_out["out.x"][0], p_out["out.y"][0], p_out["out.z"][0]])
            # print(m_out)

            m_diff = m_in-m_out
            omega = m_diff[0]
            # mom_diff_norm = np.linalg.norm(m_diff[1:4])
            mom_diff_norm = np.sqrt(m_diff[1]**2 + m_diff[2]**2 + m_diff[3]**2)
            # q = mom4_square(m_in-m_out)
            q_vals.append(mom_diff_norm)
            omega_vals.append(omega)
            mom_ins.append(m_in)
            mom_outs.append(m_out)
        
        for weight in f[key]['e/weight'].array():
            weights.append(weight)


    
    

    return np.array(mom_ins), np.array(mom_outs), np.array(q_vals), np.array(omega_vals), np.array(weights)

def mom4_square(mom4):
    return np.sqrt(mom4[0]**2 - mom4[1]**2 - mom4[2]**2 - mom4[3]**2)