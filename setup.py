import setuptools

setuptools.setup(
    name="privgem",
    version="0.1.1",
    description="privgem: Privacy-Preserving Generative Models",
    author=u"Kasra Hosseini",
    license="MIT License",
    keywords=["generative models", "privacy", "utility", "Deep Learning", "QUIPP"],
    long_description = open('README.md', encoding='utf-8', errors='replace').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    url="https://github.com/kasra-hosseini/privgem",
    download_url="https://github.com/kasra-hosseini/privgem/archive/refs/heads/develop.zip",
    packages = setuptools.find_packages(),
    include_package_data = True,
    platforms="OS Independent",
    python_requires='>=3.7,<3.9',
    install_requires=[
        "shap",
        "sdv",
        "ctgan @ git+https://github.com/kasra-hosseini/CTGAN.git",
        "opacus==0.9.0",
        "torch>=1.6.0,<2",
        "jupyterlab",
        "matplotlib",
        "PuLP",
        "networkx",
        "dython",
        "jupyter-client>=6.1.5",
        "jupyter-core>=4.6.3"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],

    entry_points={
        'console_scripts': [
            'privgem = privgem.privgem:main',
        ],
    }
)