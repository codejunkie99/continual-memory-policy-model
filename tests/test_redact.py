import unittest

from mpm.history.redact import classify, redact, scrub_sensitive, shannon_entropy


class TestRedactHostileInputs(unittest.TestCase):
    def test_rejects_email(self):
        result = redact("contact me at alice@corp.com", role="user")
        self.assertFalse(result.accept)
        self.assertIn("email", result.reason)

    def test_rejects_phone(self):
        result = redact("call me at (555) 123-4567", role="user")
        self.assertFalse(result.accept)
        self.assertIn("phone", result.reason)

    def test_rejects_street_address(self):
        result = redact("ship to 123 Main Street", role="user")
        self.assertFalse(result.accept)
        self.assertIn("address", result.reason)

    def test_rejects_credit_card(self):
        result = redact("card 4111 1111 1111 1111", role="user")
        self.assertFalse(result.accept)
        self.assertIn("card-number", result.reason)

    def test_rejects_secret_keywords(self):
        for text in ("the password is hunter2", "api_key=abc123", "my private key is here"):
            result = redact(text, role="user")
            self.assertFalse(result.accept, text)

    def test_rejects_sk_token(self):
        result = redact("use sk-1234abcd", role="user")
        self.assertFalse(result.accept)

    def test_rejects_base64(self):
        result = redact("dGhpcyBpcyBhIGJhc2U2NCBlbmNvZGVkIHNlY3JldCB2YWx1ZQ==", role="user")
        self.assertFalse(result.accept)
        self.assertEqual(result.reason, "base64")

    def test_rejects_high_entropy(self):
        token = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        self.assertTrue(shannon_entropy(token) >= 4.8)
        result = redact(token, role="user")
        self.assertFalse(result.accept)

    def test_rejects_stack_trace(self):
        trace = 'Traceback (most recent call last):\n  File "/app/main.py", line 12, in <module>\n    raise ValueError("boom")'
        self.assertFalse(redact(trace, role="user").accept)

    def test_rejects_code_blob(self):
        code = "def main():\n    import os\n    return os.getenv('KEY')"
        self.assertFalse(redact(code, role="user").accept)

    def test_rejects_url_with_secret_query(self):
        self.assertFalse(redact("see https://example.com/?token=abc123", role="user").accept)

    def test_rejects_assistant_and_system_roles(self):
        self.assertEqual(redact("I will now do X", role="assistant").reason, "assistant_output")
        self.assertEqual(redact("you are a helpful assistant", role="system").reason, "system_instruction")

    def test_rejects_empty(self):
        self.assertFalse(redact("", role="user").accept)
        self.assertFalse(redact("   ", role="user").accept)


class TestRedactScrubAndClassify(unittest.TestCase):
    def test_scrubs_absolute_paths_and_ids(self):
        text = "saved to /Users/example/notes.md and /Volumes/ExternalDisk/x.bin"
        scrubbed = scrub_sensitive(text)
        self.assertNotIn("/Users/", scrubbed)
        self.assertNotIn("/Volumes/", scrubbed)
        self.assertIn("<path>", scrubbed)

        with_id = "id 123e4567-e89b-12d3-a456-426614174000 and deadbeefdeadbeefdeadbeefdeadbeef"
        scrubbed_id = scrub_sensitive(with_id)
        self.assertNotIn("123e4567", scrubbed_id)
        self.assertNotIn("deadbeef", scrubbed_id)

    def test_accepts_durable_preference(self):
        result = redact("i prefer concise release notes", role="user")
        self.assertTrue(result.accept)
        self.assertEqual(result.category, "preference")
        self.assertEqual(result.sanitized, "i prefer concise release notes")

    def test_classify_categories(self):
        self.assertEqual(classify("we always use snake_case"), "convention")
        self.assertEqual(classify("we decided to use SQLite"), "decision")
        self.assertEqual(classify("actually that is wrong"), "correction")
        self.assertEqual(classify("the lesson was to avoid stale cache"), "lesson")
        self.assertEqual(classify("the release is in June"), "other")

    def test_default_deny_non_durable_text(self):
        result = redact("can you run the tests now", role="user")
        self.assertFalse(result.accept)
        self.assertEqual(result.reason, "not_durable")


if __name__ == "__main__":
    unittest.main()
