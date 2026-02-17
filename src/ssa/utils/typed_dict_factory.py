# Copyright (C) 2026 Adriano Lima
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from dataclasses import dataclass
from typing import Set, Tuple


@dataclass(frozen=True)
class ImmutableDict(dict):
    """A custom immutable dictionary."""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        dict.__init__(instance, *args, **kwargs)
        return instance

    def __init__(self, *args, **kwargs):
        pass  # No need to initialize here, it's done in __new__

    def __repr__(self):
        return dict.__repr__(self)

    def __str__(self):
        return dict.__str__(self)

    def __setitem__(self, key, value):
        raise TypeError("Cannot modify an immutable dictionary.")

    def __delitem__(self, key):
        raise TypeError("Cannot modify an immutable dictionary.")

    def clear(self):
        raise TypeError("Cannot modify an immutable dictionary.")

    def update(self, *args, **kwargs):
        """Raises TypeError as this dictionary is immutable.

        Args:
            \*args: Variable length argument list (not supported)
            \*\*kwargs: Arbitrary keyword arguments (not supported)

        Raises:
            TypeError: Always raises this error as the dictionary is immutable
        """
        raise TypeError("Cannot modify an immutable dictionary.")

    def pop(self, key, default=None):
        """Raises TypeError as this dictionary is immutable.

        Args:
            key: The key to remove (not supported)
            default: The default value to return if key is not found (not supported)

        Raises:
            TypeError: Always raises this error as the dictionary is immutable
        """
        raise TypeError("Cannot modify an immutable dictionary.")

    def popitem(self):
        raise TypeError("Cannot modify an immutable dictionary.")


class TypedDictFactory:
    """A class that abstracts a dictionary with a specified format."""

    def __init__(self, required_attributes: Set[Tuple[str, type]]):
        """
        Initializes the TypedDictFactory with a set of required
        attributes and their types.

        Args:
            required_attributes: An array of tuples, where each tuple contains
            the attribute name (str) and its expected type.
        """
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")
        self._logger.propagate = False

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.formatter = logging.Formatter(
            "%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
        )

        self._logger.addHandler(console_handler)

        try:
            # Checks if the required attributes follows the pattern Tuple(str, type)
            if all(
                isinstance(attr, tuple)
                and len(attr) == 2
                and isinstance(attr[0], str)
                and isinstance(attr[1], type)
                for attr in required_attributes
            ):

                self._logger.debug("Valid required attributes: %s", required_attributes)
                self._required_attributes = required_attributes

            else:
                raise AttributeError

        except AttributeError:
            self._logger.error("Invalid required attributes: %s", required_attributes)

        except Exception as e:
            self._logger.error("Error initializing the NewRelicDictFactory: %s", e)

    def __str__(self) -> str:
        try:
            return (
                "TypedDictFactory with " f"{str(self._required_attributes)} attributes"
            )
        except Exception:
            return "TypedDictFactory with no attributes"

    def __eq__(self, other):
        if isinstance(other, TypedDictFactory):
            return self._required_attributes == other._required_attributes
        return False

    def create_instance(self, **kwargs) -> ImmutableDict:
        """
        Creates an instance of the dictionary with the required attributes.

        Args:
            \*\*kwargs: Arbitrary keyword arguments representing the attributes and their values.

        Returns:
            dict: A dictionary instance with the required attributes if all checks pass, otherwise None.
        """
        try:
            # For each item in the required attributes, checks if it
            # was passed in and its type matches the expected type
            for attr, attr_type in self._required_attributes:

                self._logger.debug(
                    "Checking attribute %s with type %s",
                    attr,
                    attr_type,
                )

                if attr not in kwargs:
                    raise AttributeError(
                        "Missing required attribute: %s",
                        attr,
                    )

                if not isinstance(kwargs[attr], attr_type):
                    raise TypeError(
                        "Incorrect type for attribute %s: expected %s, got %s",
                        attr,
                        attr_type,
                        type(kwargs[attr]),
                    )

            return ImmutableDict({attr: kwargs[attr] for attr in kwargs})

        except AttributeError:
            # Inner UnboundLocalError is raised if the required attribute is
            # not a tuple, so it's not possible to be explicit in the error message
            try:
                self._logger.error(
                    "Parameter missing required attributes: (%s, %s)",
                    attr,
                    attr_type,
                )
            except UnboundLocalError:
                self._logger.error(
                    "Error creating the instance: "
                    "TypedDictFactory doesn't have a valid attributes configuration"
                )

        except TypeError:
            self._logger.error(
                "Incorrect type for attribute %s: expected %s, got %s",
                attr,
                attr_type,
                type(kwargs[attr]),
            )

        except Exception as e:
            self._logger.error(
                "Unknown error while creating the instance: %s",
                e,
            )

        return None
