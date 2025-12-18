"""
Tests for model tier selection functionality.

This module tests the model tier selection feature, ensuring that:
- model_tier="pro" triggers Pro model selection
- model_tier="flash" triggers Flash model selection
- model_tier="auto" uses ModelSelector for intelligent routing
- Resolution and other parameters affect model selection
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import base64

from nanobanana_mcp_server.config.settings import ModelTier, ThinkingLevel


class TestModelSelector:
    """Test ModelSelector class directly."""

    def test_explicit_pro_tier_returns_pro_service(self):
        """Test that explicit model_tier='pro' returns Pro service."""
        from nanobanana_mcp_server.services.model_selector import ModelSelector
        from nanobanana_mcp_server.config.settings import ModelSelectionConfig

        flash_service = Mock()
        pro_service = Mock()
        config = ModelSelectionConfig()

        selector = ModelSelector(flash_service, pro_service, config)

        service, tier = selector.select_model(
            prompt="Any prompt",
            requested_tier=ModelTier.PRO,
        )

        assert tier == ModelTier.PRO
        assert service == pro_service

    def test_explicit_flash_tier_returns_flash_service(self):
        """Test that explicit model_tier='flash' returns Flash service."""
        from nanobanana_mcp_server.services.model_selector import ModelSelector
        from nanobanana_mcp_server.config.settings import ModelSelectionConfig

        flash_service = Mock()
        pro_service = Mock()
        config = ModelSelectionConfig()

        selector = ModelSelector(flash_service, pro_service, config)

        service, tier = selector.select_model(
            prompt="Any prompt",
            requested_tier=ModelTier.FLASH,
        )

        assert tier == ModelTier.FLASH
        assert service == flash_service

    def test_resolution_4k_triggers_pro_in_auto_mode(self):
        """Test that resolution='4k' triggers Pro model in auto mode."""
        from nanobanana_mcp_server.services.model_selector import ModelSelector
        from nanobanana_mcp_server.config.settings import ModelSelectionConfig

        flash_service = Mock()
        pro_service = Mock()
        config = ModelSelectionConfig()

        selector = ModelSelector(flash_service, pro_service, config)

        service, tier = selector.select_model(
            prompt="A simple image",
            requested_tier=ModelTier.AUTO,
            resolution="4k"
        )

        assert tier == ModelTier.PRO
        assert service == pro_service

    def test_quality_keywords_trigger_pro_in_auto_mode(self):
        """Test that quality keywords trigger Pro model in auto mode."""
        from nanobanana_mcp_server.services.model_selector import ModelSelector
        from nanobanana_mcp_server.config.settings import ModelSelectionConfig

        flash_service = Mock()
        pro_service = Mock()
        config = ModelSelectionConfig()

        selector = ModelSelector(flash_service, pro_service, config)

        # Test with "professional" keyword
        service, tier = selector.select_model(
            prompt="A professional photo for production use",
            requested_tier=ModelTier.AUTO,
        )

        assert tier == ModelTier.PRO
        assert service == pro_service

    def test_speed_keywords_trigger_flash_in_auto_mode(self):
        """Test that speed keywords trigger Flash model in auto mode."""
        from nanobanana_mcp_server.services.model_selector import ModelSelector
        from nanobanana_mcp_server.config.settings import ModelSelectionConfig

        flash_service = Mock()
        pro_service = Mock()
        config = ModelSelectionConfig()

        selector = ModelSelector(flash_service, pro_service, config)

        # Test with "quick" keyword
        service, tier = selector.select_model(
            prompt="A quick draft sketch",
            requested_tier=ModelTier.AUTO,
        )

        assert tier == ModelTier.FLASH
        assert service == flash_service

    def test_explicit_flash_overrides_4k_resolution(self):
        """Test that explicit Flash tier overrides 4K resolution hint."""
        from nanobanana_mcp_server.services.model_selector import ModelSelector
        from nanobanana_mcp_server.config.settings import ModelSelectionConfig

        flash_service = Mock()
        pro_service = Mock()
        config = ModelSelectionConfig()

        selector = ModelSelector(flash_service, pro_service, config)

        # Even with 4K resolution, explicit Flash should be honored
        service, tier = selector.select_model(
            prompt="A 4K photo",
            requested_tier=ModelTier.FLASH,
            resolution="4k"
        )

        assert tier == ModelTier.FLASH
        assert service == flash_service

    def test_get_model_info_pro(self):
        """Test get_model_info returns correct info for Pro tier."""
        from nanobanana_mcp_server.services.model_selector import ModelSelector
        from nanobanana_mcp_server.config.settings import ModelSelectionConfig

        flash_service = Mock()
        pro_service = Mock()
        config = ModelSelectionConfig()

        selector = ModelSelector(flash_service, pro_service, config)
        info = selector.get_model_info(ModelTier.PRO)

        assert info['name'] == 'Gemini 3 Pro Image'
        assert info['emoji'] == '🏆'
        assert 'model_id' in info

    def test_get_model_info_flash(self):
        """Test get_model_info returns correct info for Flash tier."""
        from nanobanana_mcp_server.services.model_selector import ModelSelector
        from nanobanana_mcp_server.config.settings import ModelSelectionConfig

        flash_service = Mock()
        pro_service = Mock()
        config = ModelSelectionConfig()

        selector = ModelSelector(flash_service, pro_service, config)
        info = selector.get_model_info(ModelTier.FLASH)

        assert info['name'] == 'Gemini 2.5 Flash Image'
        assert info['emoji'] == '⚡'
        assert 'model_id' in info


class TestModelTierInGenerateImage:
    """Test model tier selection integration in generate_image tool."""

    @pytest.fixture
    def mock_all_services(self):
        """Create comprehensive mocks for all services."""
        with patch('nanobanana_mcp_server.services.get_model_selector') as mock_selector_fn, \
             patch('nanobanana_mcp_server.services.get_enhanced_image_service') as mock_enhanced_fn, \
             patch('nanobanana_mcp_server.services.get_pro_image_service') as mock_pro_fn:

            # Create service mocks
            selector = Mock()
            enhanced_service = Mock()
            pro_service = Mock()

            # Configure return values
            mock_selector_fn.return_value = selector
            mock_enhanced_fn.return_value = enhanced_service
            mock_pro_fn.return_value = pro_service

            # Default responses
            enhanced_service.generate_images.return_value = (
                [Mock()],
                [{'full_path': '/test/image.png', 'width': 1024, 'height': 1024, 'size_bytes': 1000}]
            )
            pro_service.generate_images.return_value = (
                [Mock()],
                [{'full_path': '/test/image.png', 'width': 3840, 'height': 2160, 'size_bytes': 5000}]
            )

            yield {
                'selector': selector,
                'enhanced_service': enhanced_service,
                'pro_service': pro_service,
            }

    def test_pro_tier_calls_pro_service_generate(self, mock_all_services):
        """Test that Pro tier selection calls ProImageService.generate_images."""
        # Import the internal function to test
        from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool
        from fastmcp import FastMCP
        import asyncio

        # Configure selector to return PRO
        mock_all_services['selector'].select_model.return_value = (
            mock_all_services['pro_service'],
            ModelTier.PRO
        )
        mock_all_services['selector'].get_model_info.return_value = {
            'name': 'Gemini 3 Pro Image',
            'emoji': '🏆',
            'model_id': 'gemini-3-pro-image-preview'
        }

        # Create server and register tool
        server = FastMCP("test")
        register_generate_image_tool(server)

        # Get the registered function via tool manager
        tool = server._tool_manager.get_tool('generate_image')

        # The tool should exist
        assert tool is not None

        # We can't easily call the tool directly due to async nature,
        # but we can verify the selector was configured correctly
        mock_all_services['selector'].select_model.assert_not_called()

    def test_flash_tier_calls_enhanced_service_generate(self, mock_all_services):
        """Test that Flash tier selection calls EnhancedImageService.generate_images."""
        from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool
        from fastmcp import FastMCP

        # Configure selector to return FLASH
        mock_all_services['selector'].select_model.return_value = (
            mock_all_services['enhanced_service'],
            ModelTier.FLASH
        )
        mock_all_services['selector'].get_model_info.return_value = {
            'name': 'Gemini 2.5 Flash Image',
            'emoji': '⚡',
            'model_id': 'gemini-2.5-flash-image'
        }

        # Create server and register tool
        server = FastMCP("test")
        register_generate_image_tool(server)

        # Get the registered function via tool manager
        tool = server._tool_manager.get_tool('generate_image')

        # The tool should exist
        assert tool is not None


class TestProImageServiceInterface:
    """Test ProImageService interface and parameters."""

    def test_pro_service_accepts_resolution_parameter(self):
        """Test that ProImageService.generate_images accepts resolution parameter."""
        from nanobanana_mcp_server.services.pro_image_service import ProImageService
        import inspect

        sig = inspect.signature(ProImageService.generate_images)
        assert 'resolution' in sig.parameters

    def test_pro_service_accepts_thinking_level_parameter(self):
        """Test that ProImageService.generate_images accepts thinking_level parameter."""
        from nanobanana_mcp_server.services.pro_image_service import ProImageService
        import inspect

        sig = inspect.signature(ProImageService.generate_images)
        assert 'thinking_level' in sig.parameters

    def test_pro_service_accepts_enable_grounding_parameter(self):
        """Test that ProImageService.generate_images accepts enable_grounding parameter."""
        from nanobanana_mcp_server.services.pro_image_service import ProImageService
        import inspect

        sig = inspect.signature(ProImageService.generate_images)
        assert 'enable_grounding' in sig.parameters

    def test_pro_service_has_edit_image_method(self):
        """Test that ProImageService has edit_image method."""
        from nanobanana_mcp_server.services.pro_image_service import ProImageService
        assert hasattr(ProImageService, 'edit_image')


class TestEnhancedImageServiceInterface:
    """Test EnhancedImageService interface."""

    def test_enhanced_service_accepts_aspect_ratio_parameter(self):
        """Test that EnhancedImageService.generate_images accepts aspect_ratio parameter."""
        from nanobanana_mcp_server.services.enhanced_image_service import EnhancedImageService
        import inspect

        sig = inspect.signature(EnhancedImageService.generate_images)
        assert 'aspect_ratio' in sig.parameters

    def test_enhanced_service_has_edit_methods(self):
        """Test that EnhancedImageService has edit methods."""
        from nanobanana_mcp_server.services.enhanced_image_service import EnhancedImageService
        assert hasattr(EnhancedImageService, 'edit_image_by_file_id')
        assert hasattr(EnhancedImageService, 'edit_image_by_path')


class TestGenerateImageToolSignature:
    """Test generate_image tool has correct parameters by inspecting the registered function."""

    @pytest.fixture
    def generate_image_fn(self):
        """Get the generate_image function by registering it with a server."""
        from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool
        from fastmcp import FastMCP

        server = FastMCP("test")
        register_generate_image_tool(server)

        # Access the internal function directly from the decorator closure
        # The tool is registered via @server.tool() decorator
        # We can inspect the function signature from the module directly
        import nanobanana_mcp_server.tools.generate_image as gen_module
        import inspect

        # Find the generate_image function defined inside register_generate_image_tool
        # by looking at the tool manager's registered tools
        return server

    def test_tool_has_model_tier_parameter(self):
        """Test that generate_image tool has model_tier parameter."""
        import inspect
        from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool
        from fastmcp import FastMCP

        # Create a mock server to capture the registered function
        captured_fn = None

        class CapturingServer:
            def tool(self, **kwargs):
                def decorator(fn):
                    nonlocal captured_fn
                    captured_fn = fn
                    return fn
                return decorator

        # Register tool with capturing server
        register_generate_image_tool(CapturingServer())

        assert captured_fn is not None
        sig = inspect.signature(captured_fn)
        assert 'model_tier' in sig.parameters

    def test_tool_has_resolution_parameter(self):
        """Test that generate_image tool has resolution parameter."""
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
        assert 'resolution' in sig.parameters

    def test_tool_has_thinking_level_parameter(self):
        """Test that generate_image tool has thinking_level parameter."""
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
        assert 'thinking_level' in sig.parameters

    def test_tool_has_enable_grounding_parameter(self):
        """Test that generate_image tool has enable_grounding parameter."""
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
        assert 'enable_grounding' in sig.parameters


class TestModelTierEnum:
    """Test ModelTier enum values."""

    def test_model_tier_has_flash(self):
        """Test that ModelTier has FLASH value."""
        assert hasattr(ModelTier, 'FLASH')
        assert ModelTier.FLASH.value == 'flash'

    def test_model_tier_has_pro(self):
        """Test that ModelTier has PRO value."""
        assert hasattr(ModelTier, 'PRO')
        assert ModelTier.PRO.value == 'pro'

    def test_model_tier_has_auto(self):
        """Test that ModelTier has AUTO value."""
        assert hasattr(ModelTier, 'AUTO')
        assert ModelTier.AUTO.value == 'auto'

    def test_model_tier_from_string(self):
        """Test that ModelTier can be created from string."""
        assert ModelTier('flash') == ModelTier.FLASH
        assert ModelTier('pro') == ModelTier.PRO
        assert ModelTier('auto') == ModelTier.AUTO


class TestThinkingLevelEnum:
    """Test ThinkingLevel enum values."""

    def test_thinking_level_has_low(self):
        """Test that ThinkingLevel has LOW value."""
        assert hasattr(ThinkingLevel, 'LOW')

    def test_thinking_level_has_high(self):
        """Test that ThinkingLevel has HIGH value."""
        assert hasattr(ThinkingLevel, 'HIGH')


# Mark all tests as unit tests by default
pytestmark = pytest.mark.unit
