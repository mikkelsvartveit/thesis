import unittest
import torch
import torchvision
import torchaudio


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ColoredTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        self.stream.write(
            f"{Colors.OKGREEN}✓ {test.__class__.__name__}.{test._testMethodName}{Colors.ENDC}\n"
        )

    def addError(self, test, err):
        self.stream.write(
            f"{Colors.FAIL}✗ ERROR: {test.__class__.__name__}.{test._testMethodName}{Colors.ENDC}\n"
        )
        self.stream.write(
            f"{Colors.FAIL}{self._exc_info_to_string(err, test)}{Colors.ENDC}\n"
        )

    def addFailure(self, test, err):
        self.stream.write(
            f"{Colors.FAIL}✗ FAIL: {test.__class__.__name__}.{test._testMethodName}{Colors.ENDC}\n"
        )
        self.stream.write(
            f"{Colors.FAIL}{self._exc_info_to_string(err, test)}{Colors.ENDC}\n"
        )

    def addSkip(self, test, reason):
        self.stream.write(
            f"{Colors.WARNING}⚠ SKIP: {test.__class__.__name__}.{test._testMethodName} ({reason}){Colors.ENDC}\n"
        )


class ColoredTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return ColoredTestResult(self.stream, self.descriptions, self.verbosity)


class TestPyTorchInstallation(unittest.TestCase):
    def test_pytorch_basic_imports(self):
        """Test that PyTorch and related libraries are properly installed"""
        self.assertIsNotNone(torch.__version__, "PyTorch not properly installed")
        self.assertIsNotNone(
            torchvision.__version__, "TorchVision not properly installed"
        )
        self.assertIsNotNone(
            torchaudio.__version__, "TorchAudio not properly installed"
        )
        print(f"{Colors.OKCYAN}PyTorch version: {torch.__version__}{Colors.ENDC}")

    def test_pytorch_tensor_creation(self):
        """Test basic tensor operations work"""
        x = torch.rand(5, 3)
        self.assertEqual(x.shape, (5, 3), "Failed to create tensor with correct shape")

    def test_accelerator_availability(self):
        """Test available accelerators and their functionality"""
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()

        if cuda_available:
            device = torch.device("cuda")
            device_type = "cuda"
        elif mps_available:
            device = torch.device("mps")
            device_type = "mps"
        else:
            device = torch.device("cpu")
            device_type = "cpu"

        device_info = {
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "using_device": str(device),
        }
        print(f"\n{Colors.HEADER}Device Info:{Colors.ENDC}")
        for key, value in device_info.items():
            print(f"{Colors.OKBLUE}{key}: {Colors.BOLD}{value}{Colors.ENDC}")

        if device_type == "cuda":
            print(
                f"{Colors.OKBLUE}CUDA Version: {Colors.BOLD}{torch.version.cuda}{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}GPU Device: {Colors.BOLD}{torch.cuda.get_device_name()}{Colors.ENDC}"
            )

    def test_device_tensor_operations(self):
        """Test tensor operations on the selected device"""
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)

        z = x * y
        self.assertEqual(z.device, x.device, "Operation result not on same device")
        self.assertEqual(z.shape, (3, 3), "Incorrect shape after operation")

        z = torch.matmul(x, y)
        self.assertEqual(z.device, x.device, "Matrix multiplication not on same device")
        self.assertEqual(z.shape, (3, 3), "Incorrect shape after matrix multiplication")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_specific(self):
        """Additional tests specific to CUDA devices"""
        self.assertGreater(torch.cuda.device_count(), 0, "No CUDA devices found")
        self.assertGreaterEqual(
            torch.cuda.current_device(), 0, "No current CUDA device"
        )
        self.assertTrue(torch.cuda.get_device_name(), "Could not get CUDA device name")

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
    def test_mps_specific(self):
        """Additional tests specific to MPS devices"""
        device = torch.device("mps")
        x = torch.randn(3, 3).to(device)
        self.assertEqual(
            x.device.type, "mps", "Tensor not properly moved to MPS device"
        )


if __name__ == "__main__":
    print(f"\n{Colors.HEADER}{Colors.BOLD}PyTorch Installation Test{Colors.ENDC}\n")
    runner = ColoredTestRunner(verbosity=2)
    unittest.main(testRunner=runner, verbosity=2)
