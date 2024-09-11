from typing import Any


class ConfigParser:
    @staticmethod
    def cfg_parser(cfg: dict, var_type: str, *args: str) -> list[dict]:
        """
        Parses a multi-level configuration dictionary and retrieves all entries where the value of the "var_type" key
        matches the provided var_type and, if specified, the "name" key matches any of the provided arguments.

        Args:
            cfg (dict): A multi-level configuration dictionary, potentially containing nested dictionaries and lists.
            var_type (str): The value that the "var_type" key must match.
            *args (str, optional): A variable number of strings representing the "name" values to search for. If not
                                   provided, only the "var_type" is used for matching.

        Returns:
            list[dict]: A list of dictionaries where the "var_type" matches the provided value and, if specified, the
                        "name" key matches one of the provided arguments.
        """
        result = []

        def search_in_config(node: Any) -> None:
            """
            Recursively searches through dictionaries and lists to find entries with a "var_type" key matching the given
            var_type and, if provided, a "name" key matching any of the args.

            Args:
                node (Any): The current node being traversed, which can be a dict, list, or any value.
            """
            if isinstance(node, dict):
                # Check if the dictionary has the "var_type" key and if it matches the provided var_type
                if "var_type" in node and node["var_type"] == var_type:
                    # If args are provided, also check if the "name" matches any of the args
                    if args:
                        if "name" in node and node["name"] in args:
                            result.append(node)
                    else:
                        # If no args are provided, just match by var_type
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
