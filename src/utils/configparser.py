from typing import Any, Optional


class ConfigParser:
    @staticmethod
    def cfg_parser(cfg: dict, var_type: Optional[str], *args: str) -> list[dict]:
        """
        Parses a multi-level configuration dictionary and retrieves all entries. If var_type is provided, only entries
        where the value of the "var_type" key matches the provided var_type are selected. If var_type is None, all entries
        are selected. Additionally, if specified, the "name" key must match any of the provided arguments.

        Args:
            cfg (dict): A multi-level configuration dictionary, potentially containing nested dictionaries and lists.
            var_type (str | None): The value that the "var_type" key must match. If None, all entries are matched.
            *args (str, optional): A variable number of strings representing the "name" values to search for. If not
                                   provided, only the "var_type" is used for matching.

        Returns:
            list[dict]: A list of dictionaries where the "var_type" matches the provided value or is None, and, if specified,
                        the "name" key matches one of the provided arguments.
        """
        result = []

        def search_in_config(node: Any) -> None:
            """
            Recursively searches through dictionaries and lists to find entries with a "var_type" key matching the given
            var_type (if provided) and, if specified, a "name" key matching any of the args.

            Args:
                node (Any): The current node being traversed, which can be a dict, list, or any value.
            """
            if isinstance(node, dict):
                # Check if the dictionary has the "var_type" key and if it matches the provided var_type, or var_type is None
                if "var_type" in node and (var_type is None or node["var_type"] == var_type):
                    # If args are provided, also check if the "name" matches any of the args
                    if args:
                        if "name" in node and node["name"] in args:
                            result.append(node)
                    else:
                        # If no args are provided, just match by var_type or select all if var_type is None
                        result.append(node)

                # Recursively search each key-value pair
                for value in node.values():
                    search_in_config(value)

            elif isinstance(node, list):
                # If the node is a list, search each item
                for item in node:
                    search_in_config(item)

        # Start the recursive search from the top-level config
        search_in_config(cfg)
        return result

    @staticmethod
    def find_key_in_config(cfg: dict, key: str) -> list[dict]:
        """
        Recursively searches through a multi-level configuration dictionary and retrieves all entries where the given key is present.

        Args:
            cfg (dict): A multi-level configuration dictionary.
            key (str): The key to search for in the configuration.

        Returns:
            list[dict]: A list of dictionaries where the given key is present.
        """
        result = []

        def search_for_key(node: Any) -> None:
            """
            Recursively searches through dictionaries and lists to find entries where the key is present.

            Args:
                node (Any): The current node being traversed, which can be a dict, list, or any value.
            """
            if isinstance(node, dict):
                # If the key is present in the dictionary, add the dictionary to the result
                if key in node:
                    result.append(node)

                # Recursively search each key-value pair
                for value in node.values():
                    search_for_key(value)

            elif isinstance(node, list):
                # If the node is a list, search each item
                for item in node:
                    search_for_key(item)

        # Start the recursive search from the top-level config
        search_for_key(cfg)
        return result
