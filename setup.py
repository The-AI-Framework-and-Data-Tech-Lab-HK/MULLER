# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Bingyu Liu

import logging
import os
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


project_name = "muller"
this_directory = os.path.abspath(os.path.dirname(__file__))


def build_cpp_modules():
    """Function to build cpp modules."""
    if os.getenv("BUILD_CPP", "true").lower() not in ("true", "1", "yes"):
        logging.info("Skipping C++ build because BUILD_CPP environment variable is not set to true.")
        return

    build_script = os.path.join(this_directory, "muller/util/sparsehash/build_proj.sh")

    if os.path.exists(build_script):
        logging.info("Building custom_hash_map module...")
        try:
            subprocess.run([build_script, "-j", "8"], check=True)
            logging.info("Successfully built custom_hash_map module")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to build custom_hash_map module: {e}")
            if e.returncode == 143:
                logging.warning(
                    "Build terminated (possibly due to timeout). Continuing without custom_hash_map module.")
            else:
                logging.warning(f"Build failed with code {e.returncode}. Continuing without custom_hash_map module.")


class CustomBuildPy(build_py):
    """Custom build command that builds C++ modules before building Python package."""
    def run(self):
        build_cpp_modules()
        super().run()


setup(
    cmdclass={
        'build_py': CustomBuildPy,
    },
)
