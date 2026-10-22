from pathlib import Path
import numpy as np

class IntelCSI:
    """Read raw Intel 5300 CSI ``.dat`` recordings."""

    def __init__(self, file_path: Path, nrxnum: int = 3, ntxnum: int = 1):
        self.file_path = file_path
        self.nrxnum = nrxnum
        self.ntxnum = ntxnum
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

    def read(self) -> None:
        file_size = self.file_path.stat().st_size
        max_packets = file_size // 95 + 1

        self.timestamp_low = np.zeros(max_packets, dtype=np.int64)
        self.bfee_count = np.zeros(max_packets, dtype=np.int64)
        self.Nrx = np.zeros(max_packets, dtype=np.int64)
        self.Ntx = np.zeros(max_packets, dtype=np.int64)
        self.rssi_a = np.zeros(max_packets, dtype=np.int64)
        self.rssi_b = np.zeros(max_packets, dtype=np.int64)
        self.rssi_c = np.zeros(max_packets, dtype=np.int64)
        self.noise = np.zeros(max_packets, dtype=np.int64)
        self.agc = np.zeros(max_packets, dtype=np.int64)
        self.perm = np.zeros((max_packets, 3), dtype=np.int64)
        self.rate = np.zeros(max_packets, dtype=np.int64)
        self.csi = np.zeros(
            (max_packets, 30, self.nrxnum, self.ntxnum), dtype=np.complex128
        )

        count = 0
        with self.file_path.open("rb") as stream:
            while True:
                header = stream.read(3)
                if len(header) < 3:
                    break
                field_len = (header[0] << 8) + header[1]
                code = header[2]
                if field_len < 1:
                    raise ValueError(f"Invalid packet length {field_len}")

                if code != 187:
                    stream.seek(field_len - 1, 1)
                    continue

                payload = stream.read(field_len - 1)
                if len(payload) != field_len - 1:
                    break
                if count >= max_packets:
                    raise ValueError("Packet count exceeded parser allocation")
                if len(payload) < 20:
                    raise ValueError("CSI packet payload is too short")
                if payload[8] != self.nrxnum or payload[9] != self.ntxnum:
                    raise ValueError(
                        "Unexpected antenna count: "
                        f"Nrx={payload[8]}, Ntx={payload[9]}"
                    )

                self.timestamp_low[count] = int.from_bytes(payload[0:4], "little")
                self.bfee_count[count] = int.from_bytes(payload[4:6], "little")
                self.Nrx[count] = payload[8]
                self.Ntx[count] = payload[9]
                self.rssi_a[count] = payload[10]
                self.rssi_b[count] = payload[11]
                self.rssi_c[count] = payload[12]
                self.noise[count] = int.from_bytes(
                    payload[13:14], "little", signed=True
                )
                self.agc[count] = payload[14]
                self.rate[count] = int.from_bytes(payload[18:20], "little")
                self.perm[count, 0] = payload[15] & 0x3
                self.perm[count, 1] = (payload[15] >> 2) & 0x3
                self.perm[count, 2] = (payload[15] >> 4) & 0x3

                csi_payload = payload[20:]
                index = 0
                for subcarrier in range(30):
                    index += 3
                    remainder = index & 0x7
                    for rx in range(self.nrxnum):
                        for tx in range(self.ntxnum):
                            byte_index = index // 8
                            if byte_index + 2 >= len(csi_payload):
                                raise ValueError("Truncated CSI matrix in packet")
                            real = (
                                (csi_payload[byte_index] >> remainder)
                                | (csi_payload[byte_index + 1] << (8 - remainder))
                            ) & 0xFF
                            imag = (
                                (csi_payload[byte_index + 1] >> remainder)
                                | (csi_payload[byte_index + 2] << (8 - remainder))
                            ) & 0xFF
                            if real >= 128:
                                real -= 256
                            if imag >= 128:
                                imag -= 256
                            antenna = int(self.perm[count, rx])
                            self.csi[count, subcarrier, antenna, tx] = (
                                real + 1j * imag
                            )
                            index += 16
                count += 1

        self.timestamp_low = self.timestamp_low[:count]
        self.bfee_count = self.bfee_count[:count]
        self.Nrx = self.Nrx[:count]
        self.Ntx = self.Ntx[:count]
        self.rssi_a = self.rssi_a[:count]
        self.rssi_b = self.rssi_b[:count]
        self.rssi_c = self.rssi_c[:count]
        self.noise = self.noise[:count]
        self.agc = self.agc[:count]
        self.perm = self.perm[:count]
        self.rate = self.rate[:count]
        self.csi = self.csi[:count]
        self.count = count

    @staticmethod
    def _db_inverse(value: np.ndarray) -> np.ndarray:
        return np.power(10.0, value / 10.0)

    @staticmethod
    def _db(value: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(value)

    def _total_rss(self) -> np.ndarray:
        def inverse_sometimes_zero(value: np.ndarray) -> np.ndarray:
            result = np.power(10.0, value / 10.0)
            result[result == 1] = 0
            return result

        rssi_power = (
            inverse_sometimes_zero(self.rssi_a)
            + inverse_sometimes_zero(self.rssi_b)
            + inverse_sometimes_zero(self.rssi_c)
        )
        return self._db(rssi_power) - 44 - self.agc

    def get_scaled_csi(self) -> np.ndarray:
        """Scale CSI using RSSI, thermal noise, and quantization noise."""
        csi_power = np.sum(
            (self.csi * self.csi.conj()).real, axis=(1, 2, 3)
        )
        rssi_power = self._db_inverse(self._total_rss())
        scale = rssi_power / (csi_power / 30.0)

        noise_db = self.noise.copy()
        thermal_noise_power = self._db_inverse(noise_db)
        thermal_noise_power[noise_db == -127] = self._db_inverse(
            np.array([-92.0])
        )[0]
        quantization_noise = scale * (self.Nrx * self.Ntx)
        total_noise = thermal_noise_power + quantization_noise

        scaled = self.csi * np.sqrt(scale / total_noise).reshape(-1, 1, 1, 1)
        scaled[self.Ntx == 2] *= np.sqrt(2.0)
        scaled[self.Ntx == 3] *= np.sqrt(self._db_inverse(np.array([4.5]))[0])
        return scaled.conj()



