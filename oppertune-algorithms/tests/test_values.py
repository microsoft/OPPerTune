import pytest

from oppertune.core.values import Categorical, Integer, Real, inverse_transform, transform


class TestValues:
    def test_constraints_1(self) -> None:
        """Test assertions in Constraint."""
        # Empty categories
        pytest.raises(ValueError, Categorical, name="p", val="c1", categories=())

        # Initial category not in categories
        pytest.raises(ValueError, Categorical, name="p", val="c1", categories=("c2",))

        # lb not reachable with initial value
        pytest.raises(ValueError, Integer, name="p", val=1, min=0, max=9, step=2)

        # ub not reachable with initial value
        pytest.raises(ValueError, Integer, name="p", val=100, min=0, max=750, step=100)

        # Initial value < lb
        pytest.raises(ValueError, Real, name="p", val=0.0, min=1.0, max=10.0)

        # Initial value > ub
        pytest.raises(ValueError, Real, name="p", val=11.0, min=1.0, max=10.0)

        # lb > ub
        pytest.raises(ValueError, Real, name="p", val=5.0, min=10.0, max=1.0)

    def test_constant_1(self) -> None:
        assert Categorical("p", val="c1", categories=("c1",)).is_constant
        assert Integer("p", val=1, min=1, max=1, step=1).is_constant
        assert Integer("p", val=1, min=1, max=1, step=5).is_constant
        assert Real("p", val=0.5, min=0.5, max=0.5, step=None).is_constant
        assert Real("p", val=0.5, min=0.5, max=0.5, step=0.1).is_constant

        assert not Categorical("p", val="c1", categories=("c1", "c2")).is_constant
        assert not Integer("p", val=1, min=1, max=2, step=1).is_constant
        assert not Integer("p", val=1, min=1, max=16, step=5).is_constant
        assert not Real("p", val=0.5, min=0.5, max=5.0, step=None).is_constant
        assert not Real("p", val=0.5, min=0.5, max=5.0, step=0.1).is_constant

    def test_integer_1(self) -> None:
        _min = 1
        _max = 100
        for val in range(_min, _max + 1):
            v = Integer("v", val, _min, _max)
            v_t = transform(v)
            assert type(v_t) is Integer
            assert v_t.name == v.name
            assert v_t.val == val - 1
            assert v_t.min == 0
            assert v_t.max == 99
            assert v_t.step == 1

    def test_integer_2(self) -> None:
        v = Integer("v", val=4, min=2, max=100, step=2)
        v_t = transform(v)
        assert type(v_t) is Integer
        assert v_t.name == v.name
        assert v_t.val == 1
        assert v_t.min == 0
        assert v_t.max == 49
        assert v_t.step == 1

    def test_real_1(self) -> None:
        v = Real("v", val=1, min=1, max=100)
        v_t = transform(v)
        assert type(v_t) is Real
        assert v_t.name == v.name
        assert v_t.val == 0.0
        assert v_t.min == 0.0
        assert v_t.max == 1.0
        assert v_t.step is None

    def test_real_2(self) -> None:
        v = Real("v", val=4, min=2, max=100, step=2)
        v_t = transform(v)
        assert type(v_t) is Integer
        assert v_t.name == v.name
        assert v_t.min == 0
        assert v_t.max == 49
        assert v_t.step == 1

    def test_real_3(self) -> None:
        v = Real("v", val=40, min=10, max=110)
        v_t = transform(v)
        assert type(v_t) is Real
        assert v_t.name == v.name
        assert v_t.val == 0.3
        assert v_t.min == 0.0
        assert v_t.max == 1.0
        assert v_t.step is None

    def test_conversion_1(self) -> None:
        v1 = Categorical("v1", val="c1", categories=("c1", "c2", "c3"))
        v1_t = transform(v1)
        assert type(inverse_transform(v1_t.val, v1)) is type(v1).dtype
        assert inverse_transform(v1_t.val, v1) == v1.val

        v2 = Integer("v2", val=4, min=2, max=100, step=2)
        v2_t = transform(v2)
        assert type(inverse_transform(v2_t.val, v2)) is type(v2).dtype
        assert inverse_transform(v2_t.val, v2) == v2.val
        assert inverse_transform(v2_t.min, v2) == v2.min
        assert inverse_transform(v2_t.max, v2) == v2.max

        v3 = Real("v3", val=40, min=10, max=110)
        v3_t = transform(v3)
        assert type(inverse_transform(v3_t.val, v3)) is type(v3).dtype
        assert inverse_transform(v3_t.val, v3) == v3.val
        assert inverse_transform(v3_t.min, v3) == v3.min
        assert inverse_transform(v3_t.max, v3) == v3.max

    def test_constant_conversion_1(self) -> None:
        v1 = Categorical("v1", val="c1", categories=("c1",))
        v1_t = transform(v1)
        assert v1.is_constant
        assert type(inverse_transform(v1_t.val, v1)) is type(v1).dtype
        assert inverse_transform(v1_t.val, v1) == v1.val

        v2 = Integer("v2", val=1, min=1, max=1)
        v2_t = transform(v2)
        assert v2.is_constant
        assert type(v2_t) is Integer
        assert v2_t.name == v2.name
        assert v2_t.val == 0
        assert v2_t.min == 0
        assert v2_t.max == 0
        assert v2_t.step == 1
        assert type(inverse_transform(v2_t.val, v2)) is type(v2).dtype
        assert inverse_transform(v2_t.val, v2) == v2.val
        assert inverse_transform(v2_t.min, v2) == v2.min
        assert inverse_transform(v2_t.max, v2) == v2.max

        v3 = Real("v3", val=1.5, min=1.5, max=1.5)
        v3_t = transform(v3)
        assert v3.is_constant
        assert type(v3_t) is Integer
        assert v3_t.name == v3.name
        assert v3_t.val == 0
        assert v3_t.min == 0
        assert v3_t.max == 0
        assert v3_t.step == 1
        assert type(inverse_transform(v3_t.val, v3)) is type(v3).dtype
        assert inverse_transform(v3_t.val, v3) == v3.val
        assert inverse_transform(v3_t.min, v3) == v3.min
        assert inverse_transform(v3_t.max, v3) == v3.max
