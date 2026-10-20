import itertools
import warnings

import numpy as np
import pytest

from torchwasm import torch, UnsupportedOperation


def tensor(raw, grad=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return torch.from_numpy(raw).requires_grad_(grad)


def assert_values(actual, expected, *, rtol=2e-5, atol=2e-7):
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=rtol, atol=atol)


def polynomial(raw, axis):
    """Enumerate monomials and omitted factors, independent of scan kernels."""
    coords = list(np.ndindex(raw.shape))
    values = np.zeros(raw.size)
    jacobian = np.zeros((raw.size, raw.size))
    hessians = np.zeros((raw.size, raw.size, raw.size))
    for row, out in enumerate(coords):
        factors = [i for i, point in enumerate(coords)
                   if all(point[d] == out[d] for d in range(raw.ndim) if d != axis)
                   and point[axis] <= out[axis]]
        values[row] = np.prod([raw[coords[k]] for k in factors], dtype=np.float64)
        for first in factors:
            jacobian[row, first] = np.prod([raw[coords[k]] for k in factors if k != first], dtype=np.float64)
            for second in factors:
                if first != second:
                    hessians[row, first, second] = np.prod(
                        [raw[coords[k]] for k in factors if k not in (first, second)], dtype=np.float64)
    return values, jacobian, hessians


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape,axis', [((4,),0), ((2,4),0), ((2,4),1), ((2,0,3),1), ((2,0,3),2)])
@pytest.mark.parametrize('kind', ['C', 'F', 'strided', 'readonly', 'broadcast'])
@pytest.mark.parametrize('zeros', [0, 1, 2])
@pytest.mark.parametrize('interface', ['method', 'function'])
def test_values_reverse_layout_and_zero_factors(dtype, shape, axis, kind, zeros, interface):
    raw = np.resize(np.array([1.125, -.75, 1.5, .5], dtype=dtype), shape)
    if zeros and raw.size:
        key = [slice(None)] * raw.ndim
        key[axis] = slice(0,zeros)
        raw[tuple(key)] = 0
    if kind == 'F':
        raw = np.asfortranarray(raw)
    if kind == 'strided':
        storage = np.empty(tuple(2*n for n in shape), dtype=dtype)
        view = storage[tuple(slice(None,None,2) for _ in shape)]
        view[...] = raw
        raw = view
    if kind == 'readonly':
        raw.flags.writeable = False
    if kind == 'broadcast' and raw.size:
        raw = np.broadcast_to(np.take(raw,[0],axis=axis),shape)
    before = raw.copy()
    expected, jacobian, _ = polynomial(raw,axis)
    x = tensor(raw,True)
    y = x.cumprod(axis) if interface == 'method' else torch.cumprod(x,axis)
    seed = (np.arange(raw.size,dtype=dtype).reshape(shape)+1)/4
    seed.flags.writeable = False
    gradient, = torch.autograd.grad(y,x,tensor(seed))
    assert_values(y,expected.reshape(shape))
    assert_values(gradient,(seed.reshape(-1)@jacobian).reshape(shape))
    assert y.shape == gradient.shape == shape and y.is_contiguous()
    assert x.grad is None
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('target', [torch.float32, torch.float64])
@pytest.mark.parametrize('axis', [0,1])
@pytest.mark.parametrize('point', [[1.125,-.75,1.5,.5], [0.,2.,3.,4.], [2.,0.,3.,0.]])
@pytest.mark.parametrize('mode', ['direct','functional','hvp_square','hvp_sin','hvp_norm'])
def test_jvp_and_composed_hvp_against_independent_polynomials(dtype,target,axis,point,mode):
    raw = np.array([point,point[::-1]],dtype=dtype)
    direction = np.resize(np.array([.5,-.25,.75,1.],dtype=dtype),raw.shape)
    values, jacobian, curvature = polynomial(raw,axis)
    x, v = tensor(raw,True), tensor(direction)
    f = lambda a: torch.cumprod(a,axis,dtype=target)
    if mode in ('direct','functional'):
        if mode == 'direct':
            primal, tangent = torch.func.jvp(f,(x,),(v,))
        else:
            primal, tangent = torch.autograd.functional.jvp(f,x,v)
        assert_values(primal,values.reshape(raw.shape))
        assert_values(tangent,(jacobian@direction.reshape(-1)).reshape(raw.shape))
    else:
        if mode == 'hvp_square':
            loss = lambda a: f(a).square().sum()
            expected_value = values@values
            slope, outer = 2*values, 2*np.eye(raw.size)
        elif mode == 'hvp_sin':
            loss = lambda a: f(a).sin().sum()
            expected_value = np.sin(values).sum()
            slope, outer = np.cos(values), np.diag(-np.sin(values))
        else:
            if not np.any(values):
                pytest.skip('Zero norm has a separate native boundary contract')
            loss = lambda a: torch.linalg.vector_norm(f(a))
            radius = np.linalg.norm(values)
            expected_value = radius
            slope = values/radius
            outer = np.eye(raw.size)/radius-np.outer(values,values)/radius**3
        expected_hessian = jacobian.T@outer@jacobian+np.einsum('i,ijk->jk',slope,curvature)
        primal, tangent = torch.autograd.functional.hvp(loss,x,v)
        assert_values(primal,expected_value,rtol=5e-5)
        assert_values(tangent,(expected_hessian@direction.reshape(-1)).reshape(raw.shape),rtol=1e-4,atol=2e-6)
    assert x.grad is None


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('point',[[1.125,-.75,1.5,.5],[0.,2.,3.,4.],[2.,0.,3.,0.]])
def test_differentiable_functional_jvp_has_primal_and_vector_edges(dtype,point):
    raw = np.array(point,dtype=dtype)
    values, jacobian, curvature = polynomial(raw,0)
    x = tensor(raw,True)
    direction = np.array([.5,-.25,.75,1.],dtype=dtype)
    v = tensor(direction,True)
    p, t = torch.autograd.functional.jvp(lambda a:a.cumprod(0),x,v,create_graph=True)
    seed = np.array([.25,.5,-.75,1.],dtype=dtype)
    gx, gv = torch.autograd.grad(t,(x,v),tensor(seed))
    assert_values(gx,np.einsum('i,ijk,k->j',seed,curvature,direction))
    assert_values(gv,seed@jacobian)
    assert p.requires_grad and t.requires_grad and x.grad is v.grad is None


@pytest.mark.parametrize('source,target',list(itertools.product(
    [np.float16,np.float32,np.float64],[np.float16,np.float32,np.float64])))
def test_dtype_changing_reverse_is_returned_in_input_space(source,target):
    x = tensor(np.array([2.,0.,3.,4.],dtype=source),True)
    y = x.cumprod(0,dtype=target)
    gradient, = torch.autograd.grad(y,x,tensor(np.array([1.,2.,3.,4.],dtype=target)))
    assert gradient.dtype == np.dtype(source) and y.dtype == np.dtype(target)
    assert_values(gradient,[1.,118.,0.,0.])


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_opmath_accumulation_scalar_ownership_and_saved_versions(dtype):
    values = np.resize(np.array([1+1/1024,1-1/1024,1+3/1024],dtype=dtype),129)
    accumulator = np.float32 if dtype == np.float16 else np.float64
    expected = np.multiply.accumulate(values,dtype=accumulator).astype(dtype)
    np.testing.assert_array_equal(tensor(values).cumprod(0).numpy(),expected)
    x = tensor(np.array(-0.,dtype=dtype),True)
    y = x.cumprod(-1)
    assert y is not x and not np.shares_memory(x._data,y._data)
    assert y.shape == () and np.signbit(y.item())
    y.backward(torch.tensor(2.,dtype=dtype))
    assert x.grad.item() == 2
    x = torch.ones(3,dtype=dtype,requires_grad=True)
    y = x.cumprod(0)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(UnsupportedOperation):
        y.sum().backward()


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.int16,np.int32,np.int64,np.uint64])
@pytest.mark.parametrize('shape',[(),(3,),(2,0),(2,3)])
def test_integer_default_promotion(dtype,shape):
    result = tensor(np.ones(shape,dtype=dtype)).cumprod(-1)
    assert result.dtype == torch.int64 and result.shape == shape
    np.testing.assert_array_equal(result.numpy(),np.ones(shape,dtype=np.int64))


def test_capability_no_grad_errors_and_transform_boundaries():
    assert torch._browser_capabilities['torch.cumprod'].forward
    x = torch.ones(3,requires_grad=True)
    with pytest.raises(RuntimeError):
        x.cumprod(0,dtype=torch.int64)
    with torch.no_grad():
        assert x.cumprod(0,dtype=torch.int64).dtype == torch.int64
        p, t = torch.func.jvp(lambda a:a.cumprod(0),(x,),(torch.ones_like(x),))
        assert_values(t,[1.,2.,3.])
    for axis in (True,1.,'0'):
        with pytest.raises(TypeError):
            x.cumprod(axis)
    for axis in (-2,1):
        with pytest.raises(IndexError):
            x.cumprod(axis)
    with pytest.raises(RuntimeError):
        x.cumprod(2**64)
    with pytest.raises(UnsupportedOperation):
        torch.func.jvp(lambda a:a.cumprod(0,dtype=torch.float16),(x,),(torch.ones_like(x),))
    with pytest.raises(UnsupportedOperation):
        x.cumprod(0,dtype=torch.complex64)
    assert not torch.tensor([1+2j,0j]).cumprod(0).requires_grad


@pytest.mark.parametrize('shape',[(),(1,),(2,1),(0,)])
@pytest.mark.parametrize('source,target',list(itertools.product([np.float32,np.float64],repeat=2)))
def test_single_element_scan_has_constant_functional_jacobian(shape,source,target):
    x = tensor(np.full(shape,1.125,dtype=source),True)
    v = torch.ones_like(x)
    f = lambda a:a.cumprod(-1,dtype=target).sum()
    _, h = torch.autograd.functional.hvp(f,x,v)
    np.testing.assert_array_equal(h.numpy(),np.zeros(shape,dtype=source))
    with pytest.raises(RuntimeError):
        torch.autograd.functional.hvp(f,x,v,strict=True)
    p, t = torch.autograd.functional.jvp(lambda a:a.cumprod(-1,dtype=target),x,v)
    assert_values(t,np.ones(shape,dtype=target))


@pytest.mark.parametrize('values',[[np.inf,0.,1.],[0.,np.inf,1.],[0.,np.nan,1.]])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nonfinite_zero_hvp_keeps_native_nan_result(values,dtype):
    x = tensor(np.array(values,dtype=dtype),True)
    _, h = torch.autograd.functional.hvp(lambda a:a.cumprod(0).square().sum(),x,torch.ones_like(x))
    assert np.isnan(h.numpy()).all()


@pytest.mark.parametrize('shape',[(),(1,),(65,)])
def test_dtype_overflow_is_independent_of_numpy_error_policy(shape):
    x = tensor(np.full(shape,1e300,dtype=np.float64))
    with np.errstate(all='raise'):
        y = x.cumprod(0,dtype=torch.float32)
    assert np.isposinf(y.numpy()).all()


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('length',[63,64,65,32767,32768,32769,65535,65536,65537])
@pytest.mark.parametrize('layout',['C','strided','readonly_broadcast'])
@pytest.mark.parametrize('rows',[1,3])
def test_single_working_array_and_block_boundaries_against_geometric_prefix(dtype,length,layout,rows):
    step = 1/1024
    row = np.resize(np.array([1+step,1-step],dtype=dtype),length)
    raw = np.broadcast_to(row,(rows,length)).copy()
    if layout == 'strided':
        storage = np.empty((rows,2*length),dtype=dtype)
        storage[:,::2] = raw
        raw = storage[:,::2]
    if layout == 'readonly_broadcast':
        raw = np.broadcast_to(row,(rows,length))
    before = raw.copy()
    y = tensor(raw).cumprod(1)
    # Pairing exact dyadic factors gives a closed-form geometric sequence,
    # independent of accumulation implementation and temporary-array routing.
    powers = (np.arange(length)+1)//2
    expected = (1-step*step)**powers
    expected[::2] *= 1+step
    tolerance = 1e-3 if dtype == np.float16 else 3e-6 if dtype == np.float32 else 1e-10
    assert_values(y,np.broadcast_to(expected,raw.shape),rtol=tolerance,atol=0)
    assert y.is_contiguous() and not np.shares_memory(y._data,raw)
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_zero_safe_hvp_across_short_scan_boundary(dtype):
    length, second = 66, 33
    raw = np.ones(length,dtype=dtype)
    raw[[0,second]] = 0
    x = tensor(raw,True)
    value, h = torch.autograd.functional.hvp(lambda a:a.cumprod(0).sum(),x,torch.ones_like(x),strict=True)
    # Before the second zero, H[0,k] counts prefixes k..second-1.
    # Afterwards only H[0,second] survives, with length-second prefixes.
    expected = np.zeros(length,dtype=dtype)
    expected[1:second] = np.arange(second-1,0,-1)
    expected[second] = length-second
    expected[0] = second*(second-1)//2+length-second
    assert value.item() == 0
    np.testing.assert_array_equal(h.numpy(),expected)


@pytest.mark.parametrize('shape,axis', [((8,8192),0), ((8,8192),1),
    ((8192,8),0), ((8192,8),1), ((8,32,256),0), ((8,32,256),1), ((8,32,256),2),
    ((255,256),0), ((256,256),0), ((257,256),0)])
@pytest.mark.parametrize('layout', ['C','F','strided','readonly_F','broadcast'])
@pytest.mark.parametrize('source_dtype', [np.float32,np.float64])
@pytest.mark.parametrize('zero_count', [0,1,2])
def test_large_matrix_prefix_values_and_derivatives_from_counted_monomials(shape,axis,layout,source_dtype,zero_count):
    length = shape[axis]
    first, second = length//3, 2*length//3 if zero_count == 2 else length
    row = np.ones(length,dtype=source_dtype)
    if zero_count:row[first] = 0
    if zero_count == 2:row[second] = 0
    geometry = [1]*len(shape);geometry[axis] = length
    raw = np.broadcast_to(row.reshape(geometry),shape)
    if layout != 'broadcast':
        raw = np.array(raw,order='F' if layout in ('F','readonly_F') else 'C',copy=True)
    if layout == 'strided':
        storage = np.empty(tuple(2*n for n in shape),dtype=source_dtype)
        view = storage[tuple(slice(None,None,2) for _ in shape)];view[...] = raw;raw = view
    if layout == 'readonly_F':raw.flags.writeable = False
    before = raw.copy()
    # Each prefix is a monomial. At unit factors its partial derivative is
    # one unless another factor is zero. Counting those monomials gives the
    # value, all-ones VJP and directional derivative without a scan oracle.
    values = np.ones(length,dtype=np.float64)
    reverse = np.arange(length,0,-1,dtype=source_dtype)
    tangent = np.arange(1,length+1,dtype=np.float64)
    if zero_count:
        values[first:] = 0
        reverse[first:] = 0
        reverse[:first] = np.arange(first,0,-1,dtype=source_dtype)
        reverse[first] = second-first
        tangent[first:second] = 1
        tangent[second:] = 0
    expand = lambda value: np.broadcast_to(value.reshape(geometry),shape)
    x = tensor(raw,True)
    y = x.cumprod(axis,dtype=torch.float64)
    gradient, = torch.autograd.grad(y,x,torch.ones_like(y))
    primal, direction = torch.func.jvp(lambda a:a.cumprod(axis,dtype=torch.float64),
                                     (x,),(torch.ones_like(x),))
    for actual, expected in ((y,values),(primal,values),(gradient,reverse),(direction,tangent)):
        np.testing.assert_array_equal(actual.detach().numpy(),expand(expected))
        assert actual.shape == shape
    assert y.dtype == direction.dtype == np.dtype('float64')
    assert gradient.dtype == np.dtype(source_dtype)
    assert y.is_contiguous() and not np.shares_memory(y._data,raw)
    assert x.grad is None
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('shape,axis', [((8,8192),0), ((8,8192),1),
    ((8192,8),0), ((8192,8),1), ((8,32,256),0), ((8,32,256),1), ((8,32,256),2)])
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('fill', [0.,1.,np.nan])
def test_large_negative_stride_numpy_import_keeps_native_rejection(shape,axis,dtype,fill):
    raw = np.flip(np.full(shape,fill,dtype=dtype,order='F'),axis)
    before = raw.copy()
    with pytest.raises(ValueError,match='strides'):
        tensor(raw,True)
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('shape', [(8,8192),(8,32,256)])
@pytest.mark.parametrize('readonly', [False,True])
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('zero_count', [0,1,2])
def test_wide_column_major_composed_hvp_from_monomial_hessians(shape,readonly,dtype,zero_count):
    row = np.ones(8,dtype=dtype)
    if zero_count:row[2] = 0
    if zero_count == 2:row[5] = 0
    values, jacobian, hessians = polynomial(row.reshape(8,1),0)
    expected_hessian = 2*(jacobian.T@jacobian)+np.einsum('i,ijk->jk',2*values,hessians)
    geometry = (8,)+(1,)*(len(shape)-1)
    raw = np.array(np.broadcast_to(row.reshape(geometry),shape),order='F',copy=True)
    if readonly:raw.flags.writeable = False
    before = raw.copy()
    x = tensor(raw,True)
    value, curvature = torch.autograd.functional.hvp(
        lambda a:a.cumprod(0,dtype=torch.float64).square().sum(),x,torch.ones_like(x),strict=True)
    expected = np.broadcast_to((expected_hessian@np.ones(8)).reshape(geometry),shape)
    assert value.item() == (np.prod(shape)//8)*(values@values)
    np.testing.assert_array_equal(curvature.numpy(),expected)
    assert curvature.dtype == np.dtype(dtype) and x.grad is None
    np.testing.assert_array_equal(raw,before)


def _encoded_column_case(shape,dtype,layout):
    dtype = np.dtype(dtype)
    count = int(np.prod(shape[1:]))
    indices = np.arange(count,dtype=np.int64)
    if dtype.kind == 'b':first = indices%2 == 0
    elif dtype == np.dtype('uint64'):first = (np.uint64(2**53)+2*indices.astype(np.uint64)+1)
    elif dtype == np.dtype('int64'):first = (2**53+2*indices+1)*np.where(indices%2,1,-1)
    elif dtype.kind in 'iu':first = indices%17-8
    else:first = (indices%17-8)/8
    first = np.asarray(first,dtype=dtype).reshape(shape[1:])
    raw = np.ones(shape,dtype=dtype,order='C' if layout == 'C' else 'F')
    if dtype.kind not in 'bu':raw[1:] = -1
    raw[0] = first
    if layout == 'F_strided':
        storage = np.empty(tuple(2*n for n in shape),dtype=dtype,order='F')
        view = storage[tuple(slice(None,None,2) for _ in shape)];view[...] = raw;raw = view
    if layout == 'readonly_F':raw.flags.writeable = False
    # Each output is a monomial: the distinct column tag times exact unit
    # signs. Scalar Python conversions independently pin requested rounding,
    # including the odd int64/uint64 values above2**53.
    converted = np.array([float(v) for v in first.reshape(-1)],dtype=np.float64).reshape(shape[1:])
    signs = np.ones(shape[0]) if dtype.kind in 'bu' else 1-2*(np.arange(shape[0])%2)
    expected = converted[None,...]*signs.reshape((shape[0],)+(1,)*(len(shape)-1))
    return raw,expected


@pytest.mark.parametrize('shape', [(7,5003),(3,17,1301),(5,7,11,173),(8,2048,4),
    (8,16,16,32),(8,1024,2,4),(2,32769,1,1),(1,65537)])
@pytest.mark.parametrize('dtype', [np.float16,np.float32,np.float64,np.int32,np.int64,np.uint64,np.bool_])
@pytest.mark.parametrize('layout', ['C','F','F_strided','readonly_F'])
def test_column_tags_odd_shapes_casting_and_owned_outputs(shape,dtype,layout):
    raw,expected = _encoded_column_case(shape,dtype,layout)
    before = raw.copy()
    y = tensor(raw).cumprod(0,dtype=torch.float64)
    np.testing.assert_array_equal(y.numpy(),expected)
    np.testing.assert_array_equal(np.signbit(y.numpy()),np.signbit(expected))
    assert y.dtype == np.dtype('float64') and y.is_contiguous()
    assert not np.shares_memory(y._data,raw)
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('shape', [(8,2048,4),(5,7,11,173),(256,256)])
@pytest.mark.parametrize('budget', [1024,4096,65536])
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('readonly', [False,True])
def test_column_values_independent_of_working_budget(shape,budget,dtype,readonly,monkeypatch):
    import torchwasm._reductions as reductions
    monkeypatch.setattr(reductions,'_PRODUCT_COLUMN_BYTES',budget)
    raw,expected = _encoded_column_case(shape,dtype,'readonly_F' if readonly else 'F')
    y = tensor(raw).cumprod(0,dtype=torch.float64)
    np.testing.assert_array_equal(y.numpy(),expected)
    assert y.is_contiguous() and not np.shares_memory(y._data,raw)


@pytest.mark.parametrize('rank', [8,32,64])
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_high_rank_first_axis_columns_keep_logical_positions(rank,dtype):
    shape = (2,32769)+(1,)*(rank-2)
    try:raw,expected = _encoded_column_case(shape,dtype,'F')
    except ValueError as error:
        message = str(error).lower()
        if 'maximum supported dimension' in message or 'maximum number of dimension' in message:
            pytest.skip('Installed NumPy cannot construct this rank')
        raise
    y = tensor(raw).cumprod(0,dtype=torch.float64)
    assert y.shape == shape
    np.testing.assert_array_equal(y.numpy().reshape(-1),expected.reshape(-1))
    assert y.is_contiguous() and not np.shares_memory(y._data.reshape(-1),raw.reshape(-1,order='F'))


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('layout', ['C','F','F_strided'])
@pytest.mark.parametrize('modified', ['input','output'])
def test_converted_or_packed_scan_keeps_saved_storage_guards(dtype,layout,modified):
    raw,_ = _encoded_column_case((7,5003),dtype,layout)
    x = tensor(raw,True)
    y = x.cumprod(0,dtype=torch.float64)
    with torch.no_grad():
        (x if modified == 'input' else y.detach()).add_(1)
    with pytest.raises(UnsupportedOperation,match='modified'):
        y.sum().backward()


@pytest.mark.parametrize('dtype', [np.complex64,np.complex128])
@pytest.mark.parametrize('layout', ['C','F','F_strided'])
def test_complex_to_real_conversion_keeps_warning_and_precast_values(dtype,layout):
    real,expected = _encoded_column_case((7,5003),np.float64,layout)
    raw = np.array(real+1j,dtype=dtype,order='C' if layout == 'C' else 'F')
    if layout == 'F_strided':
        storage = np.empty((14,10006),dtype=dtype,order='F');view=storage[::2,::2];view[...] = raw;raw = view
    warning_type = getattr(np,'exceptions',np).ComplexWarning
    x = tensor(raw)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        y = x.cumprod(0,dtype=torch.float64)
    assert len(observed) == 1 and issubclass(observed[0].category,warning_type)
    np.testing.assert_array_equal(y.numpy(),expected)
    assert not np.shares_memory(y._data,raw)
