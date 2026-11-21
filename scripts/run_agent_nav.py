#!/usr/bin/env python3
"""Launch a mission through the Rhizome checkout's native agent and drive nodes."""
import sys

from scene_pipeline.rhizome_agent import main


if __name__ == '__main__':
    main(['--mission-drive', *sys.argv[1:]])
