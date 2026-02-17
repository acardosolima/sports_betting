import unittest

from src.ssa.utils.typed_dict_factory import ImmutableDict, TypedDictFactory


class TestTypedDictFactory(unittest.TestCase):
    """
    Unit tests for the TypedDictFactory class.

    """

    def test_instance_immutable_dict(self):
        """
        Test creating an instance of ImmutableDict.

        This method ensures that the factory correctly creates
        an instance of ImmutableDict.
        """
        new_factory = TypedDictFactory([("country", str)])
        instance = new_factory.create_instance(country="US")

        self.assertIsInstance(instance, ImmutableDict)

    def test_immutable_dict_immutable(self):
        """
        Test creating an instance of ImmutableDict with immutable attributes.

        This method ensures that the ImmutableDict class is immutable.
        """
        new_factory = TypedDictFactory([("country", str)])
        instance = new_factory.create_instance(country="US")

        with self.assertRaises(TypeError):
            instance["country"] = "BR"

    def test_attributes_inside_list(self):
        """
        Test creating an instance with attributes defined in a list.

        This method ensures that the factory correctly creates an instance when
        the required attributes are provided as a list of tuples.
        """
        new_factory = TypedDictFactory([("country", str)])
        instance = new_factory.create_instance(country="US")

        self.assertDictEqual(instance, {"country": "US"})

    def test_additional_attributes(self):
        """
        Test creating an instance with additional attributes.

        This method ensures that the factory correctly creates an instance when
        additional attributes not defined in the factory are provided.
        """
        new_factory = TypedDictFactory([("country", str)])
        instance = new_factory.create_instance(
            country="US", vendor_id={"mondelez": "12345"}
        )

        self.assertDictEqual(
            instance, {"country": "US", "vendor_id": {"mondelez": "12345"}}
        )

    def test_multiple_instances(self):
        """
        Test creating multiple factories with different attribute values.

        This method ensures that the factory correctly creates multiple instances
        when different attribute values are provided.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", int)])

        instance1 = new_factory.create_instance(country="US", vendor_id=12345)
        instance2 = new_factory.create_instance(country="BR", vendor_id=54321)

        self.assertDictEqual(instance1, {"country": "US", "vendor_id": 12345})
        self.assertDictEqual(instance2, {"country": "BR", "vendor_id": 54321})

    def test_equal_factories(self):
        """
        Test creating multiple factories with same required attributes.

        This method ensures that the class correctly evaluate factories
        with same required attributes as equals.
        """
        new_factory1 = TypedDictFactory([("country", str), ("vendor_id", int)])
        new_factory2 = TypedDictFactory([("country", str), ("vendor_id", int)])

        self.assertEqual(new_factory1, new_factory2)

    def test_not_equal_factories(self):
        """
        Test creating multiple factories with different required attributes.

        This method ensures that the class correctly evaluate factories with
        different required attributes as not equals.
        """
        new_factory1 = TypedDictFactory([("country", str), ("vendor_id", int)])
        new_factory2 = TypedDictFactory([("state", str), ("vendor_id", int)])

        self.assertNotEqual(new_factory1, new_factory2)
        self.assertFalse(new_factory1 == new_factory2)

    def test_no_parameter(self):
        """
        Test creating an instance with no parameters.

        This method ensures that the factory correctly handles the case when
        no parameters are provided by returning None.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", str)])
        instance = new_factory.create_instance()

        self.assertIsNone(instance)

    def test_missing_parameter(self):
        """
        Test creating an instance with a missing required parameter.

        This method ensures that the factory correctly handles the case when a required
        parameter is missing by returning None.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", str)])
        instance = new_factory.create_instance(country="US")

        self.assertIsNone(instance)

    def test_invalid_parameter_type(self):
        """
        Test creating an instance with an invalid parameter type.

        This method ensures that the factory correctly handles the case when a parameter
        is provided with an invalid type by returning None.
        """
        new_factory = TypedDictFactory([("country", str)])
        instance = new_factory.create_instance(country=1)

        self.assertIsNone(instance)

    def test_missing_attribute_type(self):
        """
        Test creating an instance with a missing attribute type.

        This method ensures that the factory correctly handles the case when an attribute
        type is missing by returning None.
        """
        new_factory = TypedDictFactory([("country")])
        instance = new_factory.create_instance(country="US")

        self.assertIsNone(instance)

    def test_no_required_attribute(self):
        """
        Test creating a factory with no required attributes.

        This method ensures that the factory correctly raises a TypeError
        when no required attributes are provided.
        """
        with self.assertRaises(TypeError):
            TypedDictFactory()

    def test_args_without_keys(self):
        """
        Test creating an instance with positional arguments instead of keyword arguments.

        This method ensures that the factory correctly raises a TypeError
        when positional arguments are provided instead of keyword arguments.
        """
        with self.assertRaises(TypeError):
            new_factory = TypedDictFactory([("country", str)])
            new_factory.create_instance("US")

    def test_attributes_none_type(self):
        """
        Test creating an instance with attributes of None type.

        This method ensures that the factory correctly handles the case when attributes
        are provided with None type by returning None.
        """
        new_factory = TypedDictFactory([(None, None)])
        instance = new_factory.create_instance(country="US")

        self.assertIsNone(instance)

    def test_class_representation(self):
        """
        Test the string representation of the TypedDictFactory class.

        This method ensures that the factory correctly returns a string representation
        of itself when converted to a string.
        """
        new_factory = TypedDictFactory([("country", str)])

        self.assertEqual(
            str(new_factory),
            "TypedDictFactory with [('country', <class 'str'>)] attributes",
        )

    def test_class_representation_null_object(self):
        """
        Test the string representation of the TypedDictFactory
        class with a null object.

        This method ensures that the factory correctly raises a TypeError when
        attempting to create a factory instance without required attributes.
        """

        with self.assertRaises(Exception):
            str(TypedDictFactory())

    def test_pop_method(self):
        """
        Test the pop method of ImmutableDict.

        This method ensures that the pop method correctly raises a TypeError
        since the dictionary is immutable.
        """
        new_factory = TypedDictFactory([("country", str)])
        instance = new_factory.create_instance(country="US")

        with self.assertRaises(TypeError):
            instance.pop("country")

    def test_pop_method_with_default(self):
        """
        Test the pop method of ImmutableDict with a default value.

        This method ensures that the pop method correctly raises a TypeError
        even when a default value is provided, since the dictionary is immutable.
        """
        new_factory = TypedDictFactory([("country", str)])
        instance = new_factory.create_instance(country="US")

        with self.assertRaises(TypeError):
            instance.pop("country", "default_value")

    def test_popitem_method(self):
        """
        Test the popitem method of ImmutableDict.

        This method ensures that the popitem method correctly raises a TypeError
        since the dictionary is immutable.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", int)])
        instance = new_factory.create_instance(country="US", vendor_id=12345)

        with self.assertRaises(TypeError):
            instance.popitem()

    def test_repr_method(self):
        """
        Test the __repr__ method of ImmutableDict.

        This method ensures that the __repr__ method correctly returns
        a string representation of the dictionary.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", int)])
        instance = new_factory.create_instance(country="US", vendor_id=12345)

        expected_repr = "{'country': 'US', 'vendor_id': 12345}"
        self.assertEqual(repr(instance), expected_repr)

    def test_str_method(self):
        """
        Test the __str__ method of ImmutableDict.

        This method ensures that the __str__ method correctly returns
        a string representation of the dictionary.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", int)])
        instance = new_factory.create_instance(country="US", vendor_id=12345)

        expected_str = "{'country': 'US', 'vendor_id': 12345}"
        self.assertEqual(str(instance), expected_str)

    def test_str_method_neq(self):
        """
        Test the __str__ method of ImmutableDict.

        This method ensures that the __str__ method correctly returns
        a string representation of the dictionary.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", int)])
        instance = new_factory.create_instance(country="US", vendor_id=12345)

        incorrect_str = "{'country': 'US', 'vendor_id': 123456}"
        self.assertNotEqual(str(instance), incorrect_str)

    def test_delitem_method(self):
        """
        Test the __delitem__ method of ImmutableDict.

        This method ensures that the __delitem__ method correctly raises a TypeError
        since the dictionary is immutable.
        """
        new_factory = TypedDictFactory([("country", str)])
        instance = new_factory.create_instance(country="US")

        with self.assertRaises(TypeError):
            del instance["country"]

    def test_clear_method(self):
        """
        Test the clear method of ImmutableDict.

        This method ensures that the clear method correctly raises a TypeError
        since the dictionary is immutable.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", int)])
        instance = new_factory.create_instance(country="US", vendor_id=12345)

        with self.assertRaises(TypeError):
            instance.clear()

    def test_immutable_behavior_consistency(self):
        """
        Test that all modification methods consistently raise TypeError.

        This method ensures that all methods that could modify the dictionary
        consistently raise TypeError, maintaining the immutable behavior.
        """
        new_factory = TypedDictFactory([("country", str), ("vendor_id", int)])
        instance = new_factory.create_instance(country="US", vendor_id=12345)

        # Test all modification methods
        modification_methods = [
            lambda: instance.pop("country"),
            lambda: instance.pop("country", "default"),
            lambda: instance.popitem(),
            lambda: instance.clear(),
            lambda: instance.__delitem__("country"),
            lambda: instance.__setitem__("new_key", "new_value"),
            lambda: instance.update({"new_key": "new_value"}),
        ]

        for method in modification_methods:
            with self.assertRaises(TypeError):
                method()

        # Verify the original data is unchanged
        self.assertDictEqual(instance, {"country": "US", "vendor_id": 12345})
