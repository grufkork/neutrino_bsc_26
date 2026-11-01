import uproot
import numpy as np
from particle import Particle
import awkward as ak

# def load_events_file(path):
#     f = uproot.open(path)

#     keys = f.keys()
#     # keys = ["treeout;1"]
#     q_vals = []
#     omega_vals = []
    
#     mom_ins = []
#     mom_outs = []
    
#     weights = []

#     flagsout = []
    
#     n = 0
#     skippedcount = 0

#     for key in keys:
#         if not key.startswith("treeout;"):
#             continue
#         # for k in f[key].keys():
#         #     print(key + " " + k)
#         e_ins = f[key]['e/in'].array()
#         e_posts = f[key]['e/post'].array()
#         flags = f[key]['e/flag'].array()
#         for (e_in,e_post,flag) in zip(e_ins, e_posts, flags):
#             # for particle_idx in range(len(e["out.t"])):
#             p_in = e_in
#             p_post = e_post
#             count = len(p_post["out.t"])
#             # if count != 2:
#             #     print ("Count is " + str(count))
#             particles = ""
#             # Skip events with pions
#             skip = False
#             for pdg in p_post["post.pdg"]:
#                 if pdg in [211, -211, 111]:
#                     skippedcount += 1
#                     skip = True
#             if skip:
#                 continue
#             if n % 100 == 0:
#                 for i in range(count):
#                     particles += str(Particle.from_pdgid(p_post["post.pdg"][i]).name) + " "
#                 # print(str(n) + " " + particles)
#             n += 1

#             if flag["flag.qel"] == True:
#                 flagsout.append(0)
#             elif flag["flag.res"] == True:
#                 flagsout.append(1)
#             elif flag["flag.dis"] == True:
#                 flagsout.append(2)
#             elif flag["flag.mec"] == True:
#                 flagsout.append(3)
#             m_in = np.array([p_in["in.t"][0], p_in["in.x"][0], p_in["in.y"][0], p_in["in.z"][0]])
#             m_out = np.array([p_post["post.t"][0], p_post["post.x"][0], p_post["post.y"][0], p_post["post.z"][0]])
#             # print(m_out)

#             m_diff = m_in-m_out
#             omega = m_diff[0]
#             # mom_diff_norm = np.linalg.norm(m_diff[1:4])
#             mom_diff_norm = np.sqrt(m_diff[1]**2 + m_diff[2]**2 + m_diff[3]**2)
#             # q = mom4_square(m_in-m_out)
#             q_vals.append(mom_diff_norm)
#             omega_vals.append(omega)
#             mom_ins.append(m_in)
#             mom_outs.append(m_out)
        
#         for weight in f[key]['e/weight'].array():
#             weights.append(weight)


#     print("skipped " + str(skippedcount))
    

#     out = {
#         "mom_ins": np.array(mom_ins),
#         "mom_outs": np.array(mom_outs),
#         "q_vals": np.array(q_vals),
#         "omega_vals": np.array(omega_vals),
#         "weights": np.array(weights),
#         "mechanism": np.array(flagsout)
#     }
#     return out
    
def load_events_file(path, filter_pions=False):
    with uproot.open(path) as f:
        tree = f["treeout"]
        e_in = tree["e/in"].array()
        e_out = tree["e/out"].array()
        e_post = tree["e/post"].array()
        flags = tree["e/flag"].array()
        weights = tree["e/weight"].array()
        # print(tree.keys())

    # mom_ins = np.array([e_in["in.t"], e_in["in.x"], e_in["in.y"], e_in["in.z"]]).T
    mom_ins = ak.zip({"t": e_in["in.t"], "x": e_in["in.x"], "y": e_in["in.y"], "z": e_in["in.z"]})
    # mom_outs = np.array([e_post["out.t"], e_post["out.x"], e_post["out.y"], e_post["out.z"]]).T
    mom_outs = ak.zip({"t": e_out["out.t"], "x": e_out["out.x"], "y": e_out["out.y"], "z": e_out["out.z"]})
    mom_posts = ak.zip({"t": e_post["post.t"], "x": e_post["post.x"], "y": e_post["post.y"], "z": e_post["post.z"]})
    pdg_outs = e_post["post.pdg"]
    pdg_posts = e_post["post.pdg"]
    # has_pion = np.any(np.isin(e_out["out.pdg"], [211, -211, 111]), axis=1)
    # has_pion 

    mass_in = e_in["in._mass"]
    mass_out = e_out["out._mass"]
    mass_post = e_post["post._mass"]

    lep_in = mom_ins[:,0]
    let_out = mom_outs[:,0]

    # m_diff = lep_in - let_out
    omega_vals = lep_in.t - let_out.t
    q_x = lep_in.x - let_out.x
    q_y = lep_in.y - let_out.y
    q_z = lep_in.z - let_out.z
    q_vals = np.linalg.norm(np.array([q_x, q_y, q_z]), axis=0)

    flagsout =  0 * flags["flag.qel"] + \
                1 * flags["flag.res"] + \
                2 * flags["flag.dis"] + \
                3 * flags["flag.mec"]

    # if filter_pions:
    #     mask = ~has_pion
    #     mom_ins = mom_ins[mask]
    #     mom_outs = mom_outs[mask]
    #     q_vals = q_vals[mask]
    #     omega_vals = omega_vals[mask]
    #     weights = weights[mask]
    #     flagsout = flagsout[mask]
    #     mom_posts = mom_posts[mask]

    out = {
        "mom_ins": mom_ins[:][0],
        "mom_outs": mom_outs[:][0],
        "mom_posts": mom_posts[0],
        "q_vals": q_vals,
        "omega_vals": omega_vals,
        "weights": np.array(weights),
        "mechanism": np.array(flagsout),
        "mom_in_all": mom_ins,
        "mom_out_all": mom_outs,
        "mom_post_all": mom_posts,
        "mass_in": np.array(mass_in),
        "mass_out": mass_out,
        "mass_post": mass_post,
        "pdg_outs": pdg_outs,
        "pdg_posts": pdg_posts
    }
    return out

def mom4_square(mom4):
    return np.sqrt(mom4[0]**2 - mom4[1]**2 - mom4[2]**2 - mom4[3]**2)