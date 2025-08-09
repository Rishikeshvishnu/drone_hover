from setuptools import find_packages, setup

package_name = 'drone_img_processing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rishikesh',
    maintainer_email='rishikeshvishnusivakumar@gmail.com',
    description='Package that should take care of the cv part',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_img_coord_pub = drone_img_processing.drone_img_coord_pub:main'
        ],
    },
)
