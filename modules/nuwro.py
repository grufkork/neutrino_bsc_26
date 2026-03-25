import numpy as np

# Lifted from NuWro mecevent_2020Valencia.cc
def ConvertCoordinate_2D_to_1D(n: int, m: int) -> int:
  # Convert the 2D co-ordinates of a N x N upper triangular matrix to a 1D matrix (which stores only non zero elements)
  #formula = n*(2N -n + 1 )/2  + m-n  where N is the dimension of the upper triangular matrix 
  return ( n*(240 - n + 1)/2 + m-n)

def ConvertCoordinate_1D_to_2D(k: int) -> (int, int):
  n = np.floor(241/2 - np.sqrt((241/2)**2 - 2*k))
  m = k - n*(240 - n + 1)/2 + n
  return n, m

def TestCoordinateConversion():
  import numpy as np
  for x in np.array(range(120)):
      for y in np.array(range(120)):
          if not y > x:
              continue
          idx = ConvertCoordinate_2D_to_1D(x, y)
          (n, m) = ConvertCoordinate_1D_to_2D(idx)
          assert(x == n and y == m)

def NetworkToNuWroResponseFunctions(network: NNInterpolator) -> str:
    """
    Untested
    Takes a interpolator model and outputs response functions that can be used in NuWro
    """
    res = [""] * 5
    nrf_names = ["W00", "W03", "W11", "W12", "W33"]
    for i in range(0, 120*120):
        n, m = ConvertCoordinate_1D_to_2D(i)
        if not m > n: # q>w, Momentum transfer must be greater than energy transfer. NRFs are not defined otherwise
            continue
        q = (m+1) * 10
        w = (n+1) * 10
        val = network.predict(q, w)
        for idx in range(5):
            res[idx] += str(val[idx]) + "," # TODO format nicely, in nuwro it's a triangle
    
    for (idx, name) in enumerate(nrf_names):
        out = f"static double C_12_{name}pp[] = {{{res[idx]}}};\n"
    
    return out