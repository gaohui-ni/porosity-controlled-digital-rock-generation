import hashlib

from scripts.verify_spatial_release import content_sha256, git_lfs_oid


def test_release_hash_accepts_real_file_and_lfs_pointer(tmp_path):
    payload = tmp_path / "weights.pth"
    payload.write_bytes(b"checkpoint-bytes")
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert git_lfs_oid(payload) is None
    assert content_sha256(payload) == expected

    pointer = tmp_path / "pointer.pth"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{expected}\n"
        "size 16\n",
        encoding="utf-8",
    )
    assert git_lfs_oid(pointer) == expected
    assert content_sha256(pointer) == expected
