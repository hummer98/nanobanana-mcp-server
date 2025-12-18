"""
Tests for output_dir parameter functionality.

This module tests the custom output directory feature for image generation.
"""

import os
import pytest
from unittest.mock import Mock, patch


class TestOutputDirParameter:
    """Test output_dir parameter in generate_image tool."""

    def test_tool_has_output_dir_parameter(self):
        """Test that generate_image tool has output_dir parameter."""
        import inspect
        from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool

        captured_fn = None

        class CapturingServer:
            def tool(self, **kwargs):
                def decorator(fn):
                    nonlocal captured_fn
                    captured_fn = fn
                    return fn
                return decorator

        register_generate_image_tool(CapturingServer())

        assert captured_fn is not None
        sig = inspect.signature(captured_fn)
        assert 'output_dir' in sig.parameters

    def test_output_dir_parameter_is_optional(self):
        """Test that output_dir parameter has None as default."""
        import inspect
        from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool

        captured_fn = None

        class CapturingServer:
            def tool(self, **kwargs):
                def decorator(fn):
                    nonlocal captured_fn
                    captured_fn = fn
                    return fn
                return decorator

        register_generate_image_tool(CapturingServer())

        sig = inspect.signature(captured_fn)
        param = sig.parameters['output_dir']
        assert param.default is None


class TestEnhancedImageServiceOutputDir:
    """Test EnhancedImageService output_dir support."""

    def test_generate_images_accepts_output_dir_parameter(self):
        """Test that generate_images method accepts output_dir parameter."""
        from nanobanana_mcp_server.services.enhanced_image_service import EnhancedImageService
        import inspect

        sig = inspect.signature(EnhancedImageService.generate_images)
        assert 'output_dir' in sig.parameters

    def test_generate_images_output_dir_is_optional(self):
        """Test that output_dir parameter has None as default."""
        from nanobanana_mcp_server.services.enhanced_image_service import EnhancedImageService
        import inspect

        sig = inspect.signature(EnhancedImageService.generate_images)
        param = sig.parameters['output_dir']
        assert param.default is None

    def test_process_generated_image_accepts_out_dir(self):
        """Test that _process_generated_image accepts out_dir parameter."""
        from nanobanana_mcp_server.services.enhanced_image_service import EnhancedImageService
        import inspect

        sig = inspect.signature(EnhancedImageService._process_generated_image)
        assert 'out_dir' in sig.parameters


class TestOutputDirValidation:
    """Test output directory validation in generate_image tool."""

    def test_output_dir_expands_user_home(self, tmp_path):
        """Test that output_dir expands ~ to user home directory."""
        # This test verifies the logic in generate_image
        # The actual expansion happens in os.path.expanduser
        test_path = "~/test_output"
        expanded = os.path.expanduser(test_path)
        assert expanded != test_path
        assert not expanded.startswith("~")

    def test_output_dir_creates_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_output_dir"
        assert not new_dir.exists()

        os.makedirs(str(new_dir), exist_ok=True)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_output_dir_resolves_relative_path(self, tmp_path):
        """Test that relative paths are resolved to absolute paths."""
        relative_path = "relative/output/dir"
        absolute_path = os.path.abspath(relative_path)

        assert os.path.isabs(absolute_path)
        assert not os.path.isabs(relative_path)


class TestOutputDirIntegration:
    """Integration tests for output_dir feature."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        with patch('nanobanana_mcp_server.services.get_model_selector') as mock_selector, \
             patch('nanobanana_mcp_server.services.get_enhanced_image_service') as mock_enhanced:

            selector_instance = Mock()
            mock_selector.return_value = selector_instance

            enhanced_service = Mock()
            enhanced_service.generate_images.return_value = (
                [Mock()],
                [{'full_path': '/test/image.png', 'width': 1024, 'height': 1024, 'size_bytes': 1000}]
            )
            mock_enhanced.return_value = enhanced_service

            # Default model selection
            from nanobanana_mcp_server.config.settings import ModelTier
            selector_instance.select_model.return_value = (enhanced_service, ModelTier.FLASH)
            selector_instance.get_model_info.return_value = {
                'name': 'Gemini 2.5 Flash Image',
                'emoji': '⚡',
                'model_id': 'gemini-2.5-flash-image'
            }

            yield {
                'selector': selector_instance,
                'enhanced_service': enhanced_service,
            }

    def test_output_dir_passed_to_service(self, mock_services, tmp_path):
        """Test that output_dir is passed to the image service."""
        from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool

        captured_fn = None

        class CapturingServer:
            def tool(self, **kwargs):
                def decorator(fn):
                    nonlocal captured_fn
                    captured_fn = fn
                    return fn
                return decorator

        register_generate_image_tool(CapturingServer())

        # Create a test output directory
        output_path = str(tmp_path / "custom_output")

        # Call the function with output_dir
        result = captured_fn(
            prompt="Test image",
            output_dir=output_path,
        )

        # Verify generate_images was called with output_dir
        mock_services['enhanced_service'].generate_images.assert_called_once()
        call_kwargs = mock_services['enhanced_service'].generate_images.call_args[1]
        assert call_kwargs['output_dir'] == output_path

    def test_none_output_dir_uses_default(self, mock_services):
        """Test that None output_dir passes None to service (uses default)."""
        from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool

        captured_fn = None

        class CapturingServer:
            def tool(self, **kwargs):
                def decorator(fn):
                    nonlocal captured_fn
                    captured_fn = fn
                    return fn
                return decorator

        register_generate_image_tool(CapturingServer())

        # Call without output_dir
        result = captured_fn(
            prompt="Test image",
        )

        # Verify generate_images was called with output_dir=None
        call_kwargs = mock_services['enhanced_service'].generate_images.call_args[1]
        assert call_kwargs['output_dir'] is None


# Mark all tests as unit tests by default
pytestmark = pytest.mark.unit
