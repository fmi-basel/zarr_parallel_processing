import setuptools

def readme():
   with open('README.txt') as f:
       return f.read()

setuptools.setup(
    name = 'zarr_parallel_processing',
    version = '0.0.1',
    description = 'Scratch repository for OME-NGFF hackathon',
    long_description = readme(),
    long_description_content_type = "text/markdown",
    # license = 'MIT',
    packages = setuptools.find_packages(),
    include_package_data=True,
    )
