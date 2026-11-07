"""Exercise host transport selection against the selected Rhizome header."""
import os
from pathlib import Path
import subprocess

import pytest


def test_host_payload_identity_and_address(tmp_path):
    root=os.environ.get('RHIZOME_ROOT')
    if not root:
        pytest.skip('Set RHIZOME_ROOT for host transport checks')
    source=tmp_path/'endpoint.cpp'
    source.write_text('''#include <base_node/endpoints.hpp>
#include <iostream>
int main() {
  try {
    endpoints::Endpoints e;
    std::cout << endpoints::get_index() << " " << e.rover_eth << " " << e.payload_eth;
    return 0;
  } catch (const std::exception &) { return 2; }
}
''')
    binary=tmp_path/'endpoint'
    subprocess.run(['g++','-std=c++17','-DHOST_BUILD','-I'+str(Path(root)/'include'),str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('AIGEN_HOST_')}
    for index in ('0','1','2'):
        run=subprocess.run([str(binary)],env={**env,'AIGEN_HOST_DEVICE_INDEX':index,'AIGEN_HOST_ROVER_ADDRESS':'127.0.0.1'},capture_output=True,text=True)
        assert run.returncode==0
        assert run.stdout==f'{index} 127.0.0.1 127.0.0.1'
    for invalid in ('','-1','64','2x'):
        run=subprocess.run([str(binary)],env={**env,'AIGEN_HOST_DEVICE_INDEX':invalid},capture_output=True,text=True)
        assert run.returncode==2
