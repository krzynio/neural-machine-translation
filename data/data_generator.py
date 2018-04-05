class DataGenerator:

    def __init__(self, src_data_path, dst_data_path):
        self.src_data_path = src_data_path
        self.dst_data_path = dst_data_path
        self.src_file, self.dst_file = self._open_files()

    def next_batch(self, size):
        total = 0
        inputs = []
        outputs = []
        while total < size:
            try:
                src = next(self.src_file)
                dst = next(self.dst_file)
                inputs.append(src.replace('\n', ''))
                outputs.append(dst.replace('\n', ''))
                total += 1
            except:
                self._close_files()
                self.src_file, self.dst_file = self._open_files()
        return inputs, outputs

    def _open_files(self):
        return open(self.src_data_path), open(self.dst_data_path)

    def _close_files(self):
        self.dst_file.close()
        self.src_file.close()
