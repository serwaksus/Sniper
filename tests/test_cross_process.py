"""Integration tests for cross-process file locking and data integrity."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import file_lock, locked_update_json, save_json, load_json

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


class TestFileLock:
    def test_basic_lock_acquire_release(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        with file_lock(lock_path, timeout=5):
            assert os.path.exists(lock_path)
        with file_lock(lock_path, timeout=5):
            pass

    def test_concurrent_lock_blocks(self, tmp_path):
        """Test that two processes can't hold the same lock simultaneously."""
        lock_path = str(tmp_path / "concurrent.lock")
        script = (
            "import sys, time\n"
            "sys.path.insert(0, %r)\n"
            "from utils import file_lock\n"
            "with file_lock(%r, timeout=30):\n"
            "    time.sleep(2)\n"
            % (SRC_PATH, lock_path)
        )
        proc = subprocess.Popen([sys.executable, "-c", script])
        time.sleep(0.5)

        try:
            with file_lock(lock_path, timeout=1):
                assert False, "Should not have acquired lock"
        except TimeoutError:
            pass

        proc.wait()

    def test_locked_update_json_atomic(self, tmp_path):
        """Test that locked_update_json prevents lost updates."""
        json_path = str(tmp_path / "data.json")
        lock_dir = str(tmp_path)

        save_json(json_path, {"counter": 0})

        script = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from utils import locked_update_json\n"
            "json_path = %r\n"
            "lock_dir = %r\n"
            "for i in range(10):\n"
            "    locked_update_json(json_path, lambda d: {**d, 'counter': d.get('counter', 0) + 1}, lock_dir=lock_dir)\n"
            % (SRC_PATH, json_path, lock_dir)
        )
        proc = subprocess.Popen([sys.executable, "-c", script])

        for _ in range(10):
            locked_update_json(
                json_path,
                lambda d: {**d, "counter": d.get("counter", 0) + 1},
                lock_dir=lock_dir,
            )

        proc.wait()

        data = load_json(json_path, {})
        assert data["counter"] == 20, f"Expected 20, got {data['counter']}"


class TestPositionsIntegrity:
    """Test that positions.json survives concurrent access."""

    def test_concurrent_position_add_delete(self, tmp_path):
        positions_path = str(tmp_path / "positions.json")
        lock_dir = str(tmp_path)

        save_json(positions_path, {})

        script_add = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from utils import locked_update_json\n"
            "for i in range(10):\n"
            "    locked_update_json(\n"
            "        %r,\n"
            "        lambda d, i=i: {**d, 'slug_' + str(i): {'shares': 100, 'entry_price': 0.1}},\n"
            "        lock_dir=%r,\n"
            "    )\n"
            % (SRC_PATH, positions_path, lock_dir)
        )

        script_add2 = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from utils import locked_update_json\n"
            "for i in range(10):\n"
            "    locked_update_json(\n"
            "        %r,\n"
            "        lambda d, i=i: {**d, 'other_' + str(i): {'shares': 50, 'entry_price': 0.2}},\n"
            "        lock_dir=%r,\n"
            "    )\n"
            % (SRC_PATH, positions_path, lock_dir)
        )

        proc1 = subprocess.Popen([sys.executable, "-c", script_add])
        proc2 = subprocess.Popen([sys.executable, "-c", script_add2])
        proc1.wait()
        proc2.wait()

        data = load_json(positions_path, {})
        assert len(data) == 20, f"Expected 20 positions, got {len(data)}"
        for i in range(10):
            assert f"slug_{i}" in data
            assert f"other_{i}" in data
