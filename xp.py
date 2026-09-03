# xp.py
import numpy as np
from scipy.linalg import expm as scipy_expm
from scipy.linalg import block_diag as scipy_block_diag

import inspect
import torch
from torch.linalg import matrix_exp as torch_matrix_exp
from torch import block_diag as torch_block_diag
from dataclasses import is_dataclass, fields
from contextlib import contextmanager

class XPWrapper:
    def __init__(self, backend_type: str = 'numpy'):
        self._default_device = None
        if backend_type.lower() == 'torch':
            self.use_torch = True
            self.backend = torch
            # torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.benchmark = True

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_default_device(device)
            self._default_device = device
        elif backend_type.lower() == 'numpy':
            self.use_torch = False
            self.backend = np
        else:
            self.use_torch = False
            self.backend = np
            print(f"Unknown backend_type '{backend_type}', defaulting to numpy.")
        

    @contextmanager
    def no_grad(self):
        if self.use_torch:
            with torch.no_grad():
                yield
        else:
            yield
        
    def eye(self, *args, device=None):
        return torch.eye(*args, dtype=torch.float64, device=device or self._default_device) if self.use_torch else np.eye(*args, dtype=np.float64)

    def zeros(self, *args, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.backend.float64
        return self.backend.zeros(*args, **kwargs)
    
    def zeros_like(self, x):
        return self.backend.zeros_like(x, dtype=self.backend.float64)
    
    def inf(self, *args, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.backend.float64
        return torch.tensor(float("inf"), device=self._default_device, *args, **kwargs) if self.use_torch else np.inf
    
    def cdist(self, XA, XB, metric='euclidean'):
        if self.use_torch:
            # Torch does not have a direct cdist with metric option like scipy
            if metric == 'euclidean':
                return torch.cdist(XA, XB)
            else:
                raise NotImplementedError(f"cdist with metric '{metric}' not implemented for torch backend.")
        else:
            from scipy.spatial.distance import cdist as scipy_cdist
            return scipy_cdist(XA, XB, metric=metric)

    def ones(self, *args, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.backend.float64
        return self.backend.ones(*args, **kwargs)
    
    def linspace(self, start, stop, num=50, endpoint=True, retstep=False, dtype=None):
        if self.use_torch:
            if dtype is None:
                dtype = self.backend.float64
            return torch.linspace(start, stop, steps=num, dtype=dtype, device=self._default_device)#, device='cpu', requires_grad=False)
        else:
            return np.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype)
        
    def logspace(self, start, stop, num=50, endpoint=True, dtype=None):
        if self.use_torch:
            if dtype is None:
                dtype = self.backend.float64
            return torch.logspace(start, stop, steps=num, dtype=dtype, device=self._default_device)
        else:
            return np.logspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)
        
    def ndindex(self, *args, **kwargs):
        import itertools
        return itertools.product(*[range(s) for s in args]) if self.use_torch else np.ndindex(*args, **kwargs)
        

    def array(self, data, **kwargs):
        requires_grad = kwargs.pop('requires_grad', None)
        dtype = kwargs.pop('dtype', torch.float64)

        if self.use_torch:
            if isinstance(data, torch.Tensor):
                t = data.to(dtype=dtype, device=self._default_device)
                if requires_grad is not None:
                    if not (t.is_leaf and t.requires_grad == requires_grad):
                        t = t.clone()
                        t.requires_grad_(requires_grad)
                return t

            req = requires_grad if requires_grad is not None else False
            device = getattr(self, '_default_device', None)
            if device:
                return torch.tensor(data, dtype=dtype, requires_grad=req, device=device)
            else:
                return torch.tensor(data, dtype=dtype, requires_grad=req)

        else:
            if 'dtype' not in kwargs:
                kwargs['dtype'] = self.backend.float64
            return np.array(data, **kwargs)
        
    def isscalar(self, x):
        if self.use_torch:
            return isinstance(x, torch.Tensor) and x.ndim == 0
        else:
            return np.isscalar(x)


    def asarray(self, x, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.backend.float64       
        return torch.as_tensor(x, device=self._default_device, **kwargs) if self.use_torch else np.asarray(x, **kwargs)

    def diag(self, x):
        # Convert x to a 1-D vector
        if x.ndim == 2:
            x = x.reshape(-1)
        elif x.ndim != 1:
            raise ValueError(f"diag expects 1-D or 2-D input, got shape {x.shape}")

        if self.use_torch:
            return self.backend.diag_embed(x)
        else:
            return self.backend.diag(x)


    def reshape(self, x, shape):
        return x.reshape(shape)

    def flatten(self, x):
        return x.flatten()
    
    def clip(self, x, min_value, max_value):
        return self.backend.clip(x, min_value, max_value)

    def exp(self, x):
        return self.backend.exp(x)
    
    def sqrt(self, x):
        return self.backend.sqrt(x)
    
    def log(self, x):
        return self.backend.log(x)
    
    def log10(self,x):
        return self.backend.log10(x)
    
    def sum(self, x):
        return self.backend.sum(x)
    
    def clamp(self, x, lowerbound):
        return torch.clamp(x, min = lowerbound) if self.use_torch else np.clip(x, lowerbound)
    
    def det(self, x):
        return self.backend.linalg.det(x)

    def slogdet(self, x):
        return self.backend.linalg.slogdet(x)
    
    def solve(self, a, b):
        return self.backend.linalg.solve(a, b)
    
    def outer(self, a, b):
        return self.backend.outer(a.flatten(), b.flatten())       

    # def norm(self, x, **kwargs):
    #     return self.backend.linalg.norm(x, **kwargs) if not self.use_torch else torch.norm(x, **kwargs)
    def norm(self, x, ord=None, axis=None, keepdims=False, **kwargs):
        """
        Compute vector/matrix norm with a unified signature for numpy and torch.

        Parameters
        - x: input array/tensor
        - ord: order of the norm (maps to `p` for torch)
        - axis: axis or axes over which to compute the norm (maps to `dim` for torch)
        - keepdims: whether to keep reduced dimensions
        """
        # NumPy implementation
        if not self.use_torch:
            return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

        # Torch implementation: translate numpy arg names -> torch ones
        # torch.norm signature: torch.norm(input, p='fro', dim=None, keepdim=False, dtype=None)
        # Map ord -> p, axis -> dim, keepdims -> keepdim
        torch_kwargs = {}
        if ord is not None:
            torch_kwargs['p'] = ord
        # axis can be int, tuple or None
        torch_kwargs['dim'] = axis
        torch_kwargs['keepdim'] = keepdims
        # Pass through any other kwargs (e.g., dtype) if provided
        torch_kwargs.update(kwargs)
        return torch.norm(x, **torch_kwargs)
  
    def block(self, arrays):
        if self.use_torch:
            rows = []
            for row in arrays:
                rows.append(torch.cat(row, dim=1))  # Concatenate horizontally
            return torch.cat(rows, dim=0)  # Concatenate vertically
        else:
            return np.block(arrays)

    def block_diag(self, *arrays):       
        return torch_block_diag(*arrays) if self.use_torch else scipy_block_diag(*arrays)
    
    def expm(self, mat):
        return scipy_expm(mat) if not self.use_torch else torch_matrix_exp(mat)

    def matmul(self, a, b):
        return self.backend.matmul(a, b)

    def transpose(self, x):
        return x.T
    
    def cholesky(self, x):
        return self.backend.linalg.cholesky(x)
    
    def as_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, tuple) or isinstance(x, list):
            if isinstance(x[0], torch.Tensor):
                return np.array([xk.detach().cpu().numpy() for xk in x])
            else:
                return np.asarray(x)
        else:
            return np.asarray(x)

    def size(self, x):
        return x.numel() if self.use_torch else x.size
    
    def as_int(self, x):
        return x.int() if self.use_torch else x.astype(int)
    
    def bool(self, x):
        if isinstance(x, torch.Tensor):
            return bool(x.item())
        return bool(x)

    def convolve(self, a, v, mode='full'):
        if self.use_torch:
            # Torch does not have a direct convolve function, implement via F.conv1d
            a_reshaped = a.reshape(1, 1, -1)  # (N, C, L)
            v_reshaped = v.reshape(1, 1, -1)  # (out_channels, in_channels, kernel_size)
            convolved = torch.nn.functional.conv1d(a_reshaped, v_reshaped, padding=0)
            convolved = convolved.flatten()
            if mode == 'full':
                return convolved
            elif mode == 'valid':
                valid_length = a.shape[0] - v.shape[0] + 1
                return convolved[v.shape[0]-1: v.shape[0]-1 + valid_length]
            elif mode == 'same':
                same_length = a.shape[0]
                start = (v.shape[0] - 1) // 2
                return convolved[start: start + same_length]
            else:
                raise ValueError(f"Unknown mode '{mode}' for convolution.")
        else:
            return np.convolve(a, v, mode=mode)
    
    # def copy(self, x):
    #     """
    #     Torch backend: Always return a tensor for numeric inputs.
    #     Ensures the computation graph is preserved.
    #     """

    #     if self.use_torch:
    #         if isinstance(x, torch.Tensor):
    #             return x.clone().to(self._default_device)

    #         if isinstance(x, torch.nn.Parameter):
    #             # convert to tensor clone (Parameter.clone() not allowed)
    #             return x.detach().clone().requires_grad_(x.requires_grad).to(self._default_device)

    #         if isinstance(x, dict):
    #             return {k: self.copy(v) for k, v in x.items()}

    #         if isinstance(x, (list, tuple)):
    #             return type(x)(self.copy(v) for v in x)

    #         # For numeric types (float, int, numpy scalar) — convert to tensor
    #         if isinstance(x, (int, float)):
    #             return torch.tensor(float(x), dtype=torch.float64, device=self._default_device)

    #         # numpy array → convert to torch
    #         if isinstance(x, np.ndarray):
    #             return torch.tensor(x, dtype=torch.float64, device=self._default_device)

    #         # Unknown type → return as-is (string, None, etc.)
    #         print(f"Unknown type → return as-is: {type(x).__name__}")
    #         return x

    #     # numpy mode
    #     else:
    #         if isinstance(x, np.ndarray):
    #             return x.copy()

    #         if isinstance(x, dict):
    #             return {k: self.copy(v) for k, v in x.items()}

    #         if isinstance(x, (list, tuple)):
    #         print(f"Unknown type → return as-is: {type(x).__name__}")
    #         return x
        


    def copy(self, x):

        # ============================
        # 1️⃣ dataclass (IMU etc)
        # ============================
        if is_dataclass(x):
            cls = type(x)
            return cls(**{
                f.name: self.copy(getattr(x, f.name))
                for f in fields(x)
            })

        # ============================
        # 2️⃣ Torch backend
        # ============================
        if self.use_torch:

            # torch tensor
            if isinstance(x, torch.Tensor):
                return x.clone().to(self._default_device)

            # torch Parameter
            if isinstance(x, torch.nn.Parameter):
                return (
                    x.detach()
                    .clone()
                    .requires_grad_(x.requires_grad)
                    .to(self._default_device)
                )

            # numpy → torch
            if isinstance(x, np.ndarray):
                return torch.tensor(
                    x,
                    dtype=torch.float64,
                    device=self._default_device
                )

            # numeric → torch
            if isinstance(x, (int, float)):
                return torch.tensor(
                    float(x),
                    dtype=torch.float64,
                    device=self._default_device
                )

        # ============================
        # 3️⃣ NumPy backend
        # ============================
        else:

            # torch → numpy
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().copy()

            if isinstance(x, np.ndarray):
                return x.copy()

        # ============================
        # 4️⃣ container types
        # ============================
        if isinstance(x, dict):
            return {k: self.copy(v) for k, v in x.items()}

        if isinstance(x, (list, tuple)):
            return type(x)(self.copy(v) for v in x)

        # ============================
        # 5️⃣ others (str, None, bool…)
        # ============================
        # print(f"Unknown type → return as-is: {type(x).__name__}")
        return x



    def sin(self, x):
        return self.backend.sin(x)
    
    def cos(self, x):
        return self.backend.cos(x)
    
    def arccos(self, x):
        # torch uses arccos, numpy uses arccos
        return self.backend.arccos(x)
    
    def unwrap(self, angle, axis_in=0):
        if self.use_torch:
            tensor = torch.tensor(angle)           
            return torch.tensor(np.unwrap(tensor.cpu().numpy()))
        else:
            return np.unwrap(angle, axis=axis_in)

    def cross(self, a, b, **kwargs):
        return self.backend.cross(a, b, **kwargs)

    def dot(self, a, b, **kwargs):
        return self.backend.dot(a, b, **kwargs)

    def vstack(self, tup):
        return self.backend.vstack(tup)
    
    def hstack(self, tup):
        return self.backend.hstack(tup)
    
    def stack(self, arrays, axis=0):
        return self.backend.stack(arrays, dim=axis) if self.use_torch else self.backend.stack(arrays, axis=axis)
    
    def column_stack(self, arrays):
        return self.backend.column_stack(arrays)
    
    def full(self, shape, fill_value, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.backend.float64
        return self.backend.full(shape, fill_value, **kwargs)
    
    def newaxis(self, x):
        return x[None, :] if self.use_torch else x[np.newaxis, :]

    def diff(self, a, **kwargs):
        return self.backend.diff(a, **kwargs)

    def arange(self, *args, **kwargs):
        return self.backend.arange(*args, **kwargs)

    def mean(self, a, **kwargs):
        return self.backend.mean(a, **kwargs)

    def rand(self, *args):
        return torch.rand(*args, dtype=torch.float64) if self.use_torch else np.random.rand(*args)
        
    def arctan2(self, y, x):
        return torch.atan2(y, x) if self.use_torch else np.arctan2(y, x)
        
    def arctan(self, x):
        return self.backend.arctan(x)
    
    def arcsin(self, x):
        return self.backend.arcsin(x)   

    def floor(self, x0):
        x = self.backend.asarray(x0)
        return self.backend.floor(x)

    def ceil(self,x):
        return self.backend.ceil(x)
        
    def concatenate(self, arrays, axis=0):      
        return torch.cat(arrays, dim=axis) if self.use_torch else np.concatenate(arrays, axis=axis)
        
    def max(self, *args, **kwargs):
        return self.backend.max(*args, **kwargs)
    
    def min(self, *args, **kwargs):
        return self.backend.min(*args, **kwargs)
    
    def argmax(self, *args, **kwargs):
        return self.backend.argmax(*args, **kwargs)
    
    def argmin(self, *args, **kwargs):
        return self.backend.argmin(*args, **kwargs)
    
    def softmax(self,x, axis=-1):
        if self.use_torch:
            return torch.softmax(x, dim=axis)
        else:
            x = x - np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def where(self, condition, x=None, y=None):
        if self.use_torch:
            # torch.where requires all three arguments for element-wise selection
            if x is not None and y is not None:
                return torch.where(condition, x, y)
            else:
                return torch.where(condition)
        else:
            return np.where(condition, x, y) if x is not None and y is not None else np.where(condition)
    
    def to_backend(self, data, requires_grad=False):
        """
        Convert data to current backend (torch or numpy).
        Handles scalars, arrays, and dictionaries of arrays.
        
        Args:
            data: scalar, numpy array, torch tensor, or dict with array values
            
        Returns:
            data in current backend format
        """
        if isinstance(data, dict):
            return {k: self.to_backend(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_backend(item) for item in data)
        elif self.use_torch:
            # If already a torch Tensor, ensure dtype and optionally requires_grad
            if isinstance(data, torch.Tensor):
                t = data.to(dtype=torch.float64)
                if requires_grad:
                    if not (t.is_leaf and t.requires_grad):
                        t = t.clone().detach().to(dtype=torch.float64)
                        t.requires_grad_(True)
                return t
            # For numpy arrays or other python objects create a tensor
            return self.array(data, dtype=torch.float64, requires_grad=requires_grad)
        else:
            return np.asarray(data)
       
    @property
    def pi(self):
        return torch.tensor(torch.pi) if self.use_torch else np.pi

        
    @property
    def ndarray(self):
        return torch.Tensor if self.use_torch else np.ndarray
        
_xp = XPWrapper(backend_type='numpy')  # default to numpy


def set_backend(backend_type: str):
    global _xp
    _xp._default_device = None 
    if backend_type.lower() == 'torch':
        _xp.use_torch = True
        _xp.backend = torch 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
        _xp._default_device = device 

    elif backend_type.lower() == 'numpy':
        _xp.use_torch = False
        _xp.backend = np
    else:
        _xp.use_torch = False
        _xp.backend = np
        print(f"Unknown backend_type '{backend_type}', defaulting to numpy.")

    # Update all property proxies after backend change
    for name, prop in vars(XPWrapper).items():
        if isinstance(prop, property) and not name.startswith('_'):
            globals()[name] = getattr(_xp, name)


# Proxy methods
for name, method in inspect.getmembers(XPWrapper, predicate=inspect.isfunction):
    if not name.startswith('_'):
        globals()[name] = (lambda n: lambda *args, **kwargs: getattr(_xp, n)(*args, **kwargs))(name)

# Proxy properties (initial assignment)
for name, prop in vars(XPWrapper).items():
    if isinstance(prop, property) and not name.startswith('_'):
        globals()[name] = getattr(_xp, name)

# Make to_backend accessible at module level
def to_backend(data, requires_grad=False):
    return _xp.to_backend(data, requires_grad=requires_grad)
