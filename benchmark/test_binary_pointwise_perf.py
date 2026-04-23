from typing import Generator

import pytest
import torch

from benchmark.attri_util import (
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    INT_DTYPES,
)
from benchmark.performance_utils import Benchmark, Config, generate_tensor_input


class BinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)
        shape2 = list(args[0].shape)
        return torch.tensor(shape1).prod().item() + torch.tensor(shape2).prod().item()


class GcdBenchmark(BinaryPointwiseBenchmark):
    # gcd has data-dependent iteration counts, so memory throughput is more
    # interpretable than a synthetic tflops number here.
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["gbps"]

    def get_gbps(self, args, latency):
        numel = args[0].numel()
        bytes_moved = numel * (args[0].element_size() + args[1].element_size())
        bytes_moved += numel * torch.empty((), dtype=args[0].dtype).element_size()
        return bytes_moved / latency * 1e-6


def _make_power_of_two_tensor(shape, dtype, device):
    max_exp = torch.iinfo(dtype).bits - 2
    exponents = torch.randint(0, max_exp + 1, shape, device=device, dtype=torch.int64)
    return torch.bitwise_left_shift(
        torch.ones(shape, dtype=dtype, device=device), exponents.to(dtype)
    )


class GcdDistributionBenchmark(GcdBenchmark):
    DISTRIBUTIONS = ("random", "equal", "unit", "power_of_two", "multiples")
    DISTRIBUTION_SHAPES = [(64, 64), (4096, 4096), (64, 512, 512)]
    _current_distribution = None

    def init_default_config(self):
        self.shapes = self.DISTRIBUTION_SHAPES

    def init_user_config(self):
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.shapes = self.DISTRIBUTION_SHAPES

    def _make_distribution_pair(self, shape, cur_dtype, distribution):
        lhs = generate_tensor_input(shape, cur_dtype, self.device)
        rhs = generate_tensor_input(shape, cur_dtype, self.device)

        if distribution == "random":
            return lhs, rhs
        if distribution == "equal":
            return lhs, lhs.clone()
        if distribution == "unit":
            lhs = torch.where(lhs >= 0, 1, -1).to(cur_dtype)
            return lhs, rhs
        if distribution == "power_of_two":
            return (
                _make_power_of_two_tensor(shape, cur_dtype, self.device),
                _make_power_of_two_tensor(shape, cur_dtype, self.device),
            )
        if distribution == "multiples":
            base = torch.randint(1, 32, shape, dtype=cur_dtype, device=self.device)
            lhs_scale = torch.randint(1, 16, shape, dtype=cur_dtype, device=self.device)
            rhs_scale = torch.randint(1, 16, shape, dtype=cur_dtype, device=self.device)
            return base * lhs_scale, base * rhs_scale
        raise ValueError(f"unsupported gcd distribution: {distribution}")

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            for distribution in self.DISTRIBUTIONS:
                lhs, rhs = self._make_distribution_pair(shape, cur_dtype, distribution)
                yield lhs, rhs, {"distribution": distribution}

    def unpack_to_args_kwargs(self, input_tuple):
        args, kwargs = super().unpack_to_args_kwargs(input_tuple)
        self._current_distribution = kwargs.pop("distribution", None)
        return args, kwargs

    def record_shapes(self, *args, **kwargs):
        recorded = super().record_shapes(*args, **kwargs)
        distribution = self._current_distribution
        if distribution is None:
            return recorded
        if isinstance(recorded, list):
            return recorded + [f"distribution={distribution}"]
        return [recorded, f"distribution={distribution}"]


class GcdOutBenchmark(GcdBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            lhs = generate_tensor_input(shape, cur_dtype, self.device)
            rhs = generate_tensor_input(shape, cur_dtype, self.device)
            out = torch.empty_like(lhs)
            yield lhs, rhs, {"out": out}


class ScalarBinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations with scalar input.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = 0.001  # Scalar input
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[1].shape)  # Second argument is the tensor
        return torch.tensor(shape).prod().item()


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in [
            ("add", torch.add, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("allclose", torch.allclose, FLOAT_DTYPES + INT_DTYPES),
            ("bitwise_and", torch.bitwise_and, INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES),
            ("div", torch.div, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("dunder_or", lambda a, b: a | b, INT_DTYPES + BOOL_DTYPES),
            ("eq", torch.eq, FLOAT_DTYPES),
            ("equal", torch.equal, FLOAT_DTYPES),
            ("floor_divide", torch.floor_divide, INT_DTYPES),
            ("fmin", torch.fmin, FLOAT_DTYPES),
            ("gcd", torch.gcd, INT_DTYPES),
            ("ge", torch.ge, FLOAT_DTYPES),
            ("greater", torch.greater, FLOAT_DTYPES),
            ("gt", torch.gt, FLOAT_DTYPES),
            ("hypot", torch.hypot, FLOAT_DTYPES),
            ("isclose", torch.isclose, FLOAT_DTYPES + INT_DTYPES),
            ("le", torch.le, FLOAT_DTYPES),
            ("logaddexp", torch.logaddexp, FLOAT_DTYPES),
            ("logical_and", torch.logical_and, INT_DTYPES + BOOL_DTYPES),
            ("logical_or", torch.logical_or, INT_DTYPES + BOOL_DTYPES),
            ("logical_xor", torch.logical_xor, INT_DTYPES + BOOL_DTYPES),
            ("lt", torch.lt, FLOAT_DTYPES),
            ("maximum", torch.maximum, FLOAT_DTYPES),
            ("minimum", torch.minimum, FLOAT_DTYPES),
            ("mul", torch.mul, FLOAT_DTYPES + COMPLEX_DTYPES),
            ("ne", torch.ne, FLOAT_DTYPES),
            ("polar", torch.polar, [torch.float32]),
            ("pow", torch.pow, FLOAT_DTYPES),
            ("remainder", torch.remainder, INT_DTYPES),
            ("sub", torch.sub, FLOAT_DTYPES + COMPLEX_DTYPES),
        ]
    ],
)
def test_general_binary_pointwise(op_name, torch_op, dtypes):
    benchmark_cls = GcdBenchmark if op_name == "gcd" else BinaryPointwiseBenchmark
    bench = benchmark_cls(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


@pytest.mark.gcd
def test_gcd_distribution_binary_pointwise():
    bench = GcdDistributionBenchmark(
        op_name="gcd_distribution",
        torch_op=torch.gcd,
        dtypes=INT_DTYPES,
    )
    bench.run()


@pytest.mark.gcd
def test_gcd_out_binary_pointwise():
    bench = GcdOutBenchmark(
        op_name="gcd_out",
        torch_op=torch.gcd,
        dtypes=INT_DTYPES,
    )
    bench.run()


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in [
            ("add_", lambda a, b: a.add_(b), FLOAT_DTYPES),
            ("bitwise_and_", lambda a, b: a.bitwise_and_(b), INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or_", lambda a, b: a.bitwise_or_(b), INT_DTYPES + BOOL_DTYPES),
            ("div_", lambda a, b: a.div_(b), FLOAT_DTYPES),
            ("dunder_ior", lambda a, b: a.__ior__(b), INT_DTYPES + BOOL_DTYPES),
            ("floor_divide_", lambda a, b: a.floor_divide_(b), INT_DTYPES),
            ("logical_and_", lambda a, b: a.logical_and_(b), INT_DTYPES + BOOL_DTYPES),
            ("logical_or_", lambda a, b: a.logical_or_(b), INT_DTYPES + BOOL_DTYPES),
            ("mul_", lambda a, b: a.mul_(b), FLOAT_DTYPES),
            ("pow_", lambda a, b: a.pow_(b), FLOAT_DTYPES),
            ("remainder_", lambda a, b: a.remainder_(b), INT_DTYPES),
            ("sub_", lambda a, b: a.sub_(b), FLOAT_DTYPES),
        ]
    ],
)
def test_general_inplace_binary_pointwise(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(
        op_name=op_name, torch_op=torch_op, dtypes=dtypes, is_inplace=True
    )
    bench.run()


@pytest.mark.pow
def test_pow(op_name, torch_op, dtypes):
    bench = ScalarBinaryPointwiseBenchmark(
        op_name="pow",
        torch_op=lambda a, b: torch.pow(a, b),
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
