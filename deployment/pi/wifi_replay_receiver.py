import argparse
import hashlib
import os
import socket
from pathlib import Path

EXPECTED = {
    "P": (
        "POS102.BIN",
        "05494a672cc8fada857006410dc036f9235dd042caaf61256e0e1bb48131db71",
    ),
    "N": (
        "NEG024.BIN",
        "e7d2360f128ddf5fbda2c00eb3633f5ebe246ba63ee12227de78fb290ca1db6d",
    ),
}
EXPECTED_BYTES = 600_000
OUTPUT = Path.home() / "afib_replay" / "captures"


def recv_line(conn, limit=100):
    data = bytearray()
    while len(data) < limit:
        byte = conn.recv(1)
        if not byte:
            raise ConnectionError("ESP32 disconnected while sending header")
        if byte == b"\n":
            return data.decode("ascii").strip()
        data.extend(byte)
    raise ValueError("Header too long")


def receive(choice):
    filename, expected_hash = EXPECTED[choice]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    final_path = OUTPUT / filename
    partial_path = OUTPUT / (filename + ".part")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 8766))
        server.listen(1)
        print(f"WAITING for ESP32 on port 8766; requesting {filename}", flush=True)

        conn, address = server.accept()
        with conn:
            conn.settimeout(20)
            print(f"Connected from {address[0]}", flush=True)
            try:
                greeting = recv_line(conn)
                if greeting != "READY":
                    raise ValueError(f"Unexpected greeting: {greeting!r}")

                conn.sendall((choice + "\n").encode("ascii"))
                header = recv_line(conn)
                expected_header = f"DATA {choice} {EXPECTED_BYTES}"
                if header != expected_header:
                    raise ValueError(f"Unexpected header: {header!r}")

                digest = hashlib.sha256()
                remaining = EXPECTED_BYTES
                with partial_path.open("wb") as output:
                    while remaining:
                        chunk = conn.recv(min(8192, remaining))
                        if not chunk:
                            raise ConnectionError(
                                f"ESP32 disconnected with {remaining} bytes missing"
                            )
                        output.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                        received = EXPECTED_BYTES - remaining
                        if received % 30_000 < len(chunk):
                            print(f"Received {received}/{EXPECTED_BYTES}", flush=True)
                    output.flush()
                    os.fsync(output.fileno())

                actual_hash = digest.hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"SHA-256 mismatch: got {actual_hash}, expected {expected_hash}"
                    )

                os.replace(partial_path, final_path)
                conn.sendall(b"PASS\n")
                print(f"PASS: {final_path}", flush=True)
                print(f"Bytes: {EXPECTED_BYTES}; SHA-256: {actual_hash}", flush=True)

            except Exception as exc:
                partial_path.unlink(missing_ok=True)
                try:
                    conn.sendall(b"FAIL\n")
                except OSError:
                    pass
                print(f"FAIL: {exc}", flush=True)
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("choice", choices=["P", "N"])
    receive(parser.parse_args().choice)
