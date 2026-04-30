"""Tests for the EveryVoice ↔ StyleTTS2 config/text integration."""

import unittest

from pydantic import ValidationError


class TestStyleTTS2PretrainedConfig(unittest.TestCase):
    def test_default_pretrained_symbols_length(self):
        from styletts2.ev_config import StyleTTS2PretrainedConfig

        cfg = StyleTTS2PretrainedConfig()
        self.assertEqual(len(cfg.pretrained_symbols), 178)

    def test_default_pretrained_symbols_first_three(self):
        from styletts2.ev_config import StyleTTS2PretrainedConfig
        from styletts2.text_utils import symbols

        cfg = StyleTTS2PretrainedConfig()
        self.assertEqual(cfg.pretrained_symbols[:3], list(symbols)[:3])


class TestStyleTTS2ModelConfig(unittest.TestCase):
    def test_default_target_text_representation_level(self):
        from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel

        from styletts2.ev_config import StyleTTS2ModelConfig

        cfg = StyleTTS2ModelConfig()
        self.assertEqual(
            cfg.target_text_representation_level,
            TargetTrainingTextRepresentationLevel.characters,
        )


_CONTACT = {"contact": {"contact_name": "Test", "contact_email": "test@test.com"}}


class TestToNativeConfig(unittest.TestCase):
    def _make_config(self, **model_kwargs):
        from everyvoice.config.text_config import TextConfig

        from styletts2.ev_config import StyleTTS2Config

        text = TextConfig()
        return StyleTTS2Config(text=text, **_CONTACT, **model_kwargs)

    def test_target_text_representation_in_data_params_characters(self):
        from styletts2.ev_config.translation import to_native_config

        cfg = self._make_config()
        native = to_native_config(cfg)
        self.assertEqual(native["data_params"]["target_text_representation"], "characters")

    def test_target_text_representation_in_data_params_phones(self):
        from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel

        from styletts2.ev_config import StyleTTS2ModelConfig
        from styletts2.ev_config.translation import to_native_config

        cfg = self._make_config(
            model=StyleTTS2ModelConfig(
                target_text_representation_level=TargetTrainingTextRepresentationLevel.ipa_phones
            )
        )
        native = to_native_config(cfg)
        # TargetTrainingTextRepresentationLevel.ipa_phones.value == "phones"
        self.assertEqual(native["data_params"]["target_text_representation"], "phones")


class TestEVStyleTTS2TextEncoder(unittest.TestCase):
    def setUp(self):
        # _WARN_ONCE is module-level state; clear it so warning tests are independent.
        import styletts2.ev_config.text as _text_mod
        _text_mod._WARN_ONCE.clear()

    def _make_encoder(self):
        from everyvoice.config.text_config import TextConfig

        from styletts2.ev_config import StyleTTS2PretrainedConfig
        from styletts2.ev_config.text import EVStyleTTS2TextEncoder

        symbols = StyleTTS2PretrainedConfig().pretrained_symbols
        return EVStyleTTS2TextEncoder(TextConfig(), symbols), symbols

    def test_encode_character_tokens(self):
        encoder, symbols = self._make_encoder()
        # "h/e/l/l/o" — these are all in the pretrained Latin+IPA set
        # Use simple latin chars that are definitely in the pretrained set
        indices = encoder.encode_token_sequence("h/e/l/l/o")
        self.assertEqual(len(indices), 5)
        # Each index should be a valid position in the symbol table
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(symbols))

    def test_encode_ipa_phone_tokens(self):
        encoder, symbols = self._make_encoder()
        # IPA phones that are in StyleTTS2's _letters_ipa set
        indices = encoder.encode_token_sequence("h/ɛ/l/o/ʊ")
        self.assertEqual(len(indices), 5)
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(symbols))

    def test_punctuation_remapping_tokens(self):
        from everyvoice.text.features import DEFAULT_PUNCTUATION_HASH

        encoder, symbols = self._make_encoder()
        tokens = {DEFAULT_PUNCTUATION_HASH["exclamations"]: "!", DEFAULT_PUNCTUATION_HASH["commas"]: ","}
        for k, v in tokens.items():
            indices = encoder.encode_token_sequence(k)
            self.assertEqual(len(indices), 1)
            # "!" and "," should be in the pretrained symbol table
            self.assertEqual(symbols[indices[0]], v)

    def test_paren_dropped_with_warning(self, caplog=None):
        from everyvoice.text.features import DEFAULT_PUNCTUATION_HASH

        encoder, _ = self._make_encoder()
        paren_token = DEFAULT_PUNCTUATION_HASH["parentheses"]
        # <PAREN> has no StyleTTS2 equivalent — should be silently dropped
        with self.assertLogs("styletts2.ev_config.text", level="WARNING") as cm:
            indices = encoder.encode_token_sequence(paren_token)
        self.assertEqual(indices, [])
        self.assertTrue(any("no mapping" in msg for msg in cm.output))

    def test_unknown_token_dropped_with_warning(self):
        encoder, _ = self._make_encoder()
        with self.assertLogs("styletts2.ev_config.text", level="WARNING") as cm:
            indices = encoder.encode_token_sequence("<UNKNOWN_TOKEN_XYZ>")
        self.assertEqual(indices, [])
        self.assertTrue(any("no mapping" in msg for msg in cm.output))

    def test_empty_token_sequence(self):
        encoder, _ = self._make_encoder()
        indices = encoder.encode_token_sequence("")
        self.assertEqual(indices, [])

    def test_mixed_valid_and_dropped_tokens(self):
        from everyvoice.text.features import DEFAULT_PUNCTUATION_HASH

        encoder, symbols = self._make_encoder()
        paren_token = DEFAULT_PUNCTUATION_HASH["parentheses"]
        # "h / <PAREN> / e" — PAREN should be dropped, h and e kept
        with self.assertLogs("styletts2.ev_config.text", level="WARNING"):
            indices = encoder.encode_token_sequence(f"h/{paren_token}/e")
        self.assertEqual(len(indices), 2)
        self.assertEqual(symbols[indices[0]], "h")
        self.assertEqual(symbols[indices[1]], "e")


class TestSymbolSubsetValidator(unittest.TestCase):
    def _text_config_with_extra_symbols(self, *extra: str):
        """Return a TextConfig with *extra* symbols added as a custom symbol set."""
        from everyvoice.config.text_config import Symbols, TextConfig

        # Symbols uses extra="allow" — arbitrary keyword args become extra fields.
        symbols = Symbols(custom_letters=list(extra))
        return TextConfig(symbols=symbols)

    def test_validator_rejects_unknown_symbol(self):
        """A symbol not in the pretrained table should raise ValidationError."""
        from styletts2.ev_config import StyleTTS2Config

        # Korean character definitely not in StyleTTS2's Latin+IPA table
        bad_text = self._text_config_with_extra_symbols("가")
        with self.assertRaises(ValidationError) as ctx:
            StyleTTS2Config(text=bad_text, **_CONTACT)
        self.assertIn("가", str(ctx.exception))

    def test_validator_rejects_diphthong_not_in_pretrained(self):
        """A multi-character diphthong like 'oʊ' is not in the pretrained table."""
        from styletts2.ev_config import StyleTTS2Config

        bad_text = self._text_config_with_extra_symbols("oʊ")
        with self.assertRaises(ValidationError):
            StyleTTS2Config(text=bad_text, **_CONTACT)

    def test_validator_accepts_valid_ipa_phones(self):
        """Single IPA phones in StyleTTS2's _letters_ipa set pass validation."""
        from styletts2.ev_config import StyleTTS2Config

        # "ɛ" and "ɝ" are both in StyleTTS2's pretrained _letters_ipa
        good_text = self._text_config_with_extra_symbols("ɛ", "ɝ")
        StyleTTS2Config(text=good_text, **_CONTACT)  # should not raise

    def test_validator_accepts_default_text_config(self):
        """The default TextConfig (no declared letters) passes validation trivially."""
        from everyvoice.config.text_config import TextConfig

        from styletts2.ev_config import StyleTTS2Config

        StyleTTS2Config(text=TextConfig(), **_CONTACT)
