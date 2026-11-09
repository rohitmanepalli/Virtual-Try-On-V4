import unittest
from app.models import BodyProfile
from app.services.session_manager import SessionManager, SessionRecord


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager()
        self.sample_profile = BodyProfile(
            height_ratio=0.82,
            shoulder_width_ratio=0.28,
            torso_aspect_ratio=1.14,
            shoulder_to_hip_ratio=1.22,
            chest_proportion=1.05,
            waist_proportion=0.84,
            hip_proportion=1.0,
            body_type="Athletic / V-Shape",
            confidence_score=0.92,
            frame_count_sampled=6
        )

    def test_ephemeral_key_generated_per_session(self):
        """Verify each session receives a unique, fresh Fernet key."""
        s1 = self.manager.get_or_create_session("session_alpha")
        s2 = self.manager.get_or_create_session("session_beta")

        self.assertIsNotNone(s1._key)
        self.assertIsNotNone(s2._key)
        self.assertNotEqual(s1._key, s2._key, "Fernet keys must be unique per session, not shared globally.")

    def test_zero_retention_encryption_and_retrieval(self):
        """Verify profile is stored as ciphertext in-memory and decrypts correctly."""
        session = self.manager.get_or_create_session("session_gamma")
        session.set_body_profile(self.sample_profile)

        # In-memory buffer is ciphertext, not plaintext
        self.assertIsInstance(session._encrypted_profile, bytes)
        self.assertNotIn(b"Athletic / V-Shape", session._encrypted_profile)

        # Retrieve and verify decrypted profile
        retrieved = session.get_body_profile()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.body_type, "Athletic / V-Shape")
        self.assertAlmostEqual(retrieved.shoulder_width_ratio, 0.28)

    def test_cryptographic_erasure_on_purge(self):
        """Verify purging zeroes the key and ciphertext and removes the session."""
        session = self.manager.get_or_create_session("session_delta")
        session.set_body_profile(self.sample_profile)
        session.set_last_good_render("data:image/jpeg;base64,sample_render_data")
        import numpy as np
        sample_mask = np.zeros((100, 80), dtype=np.float32)
        sample_mask[10:30, 20:60] = 1.0
        session.set_face_mask(sample_mask)

        purged = self.manager.purge_session("session_delta")
        self.assertTrue(purged)
        self.assertIsNone(self.manager.get_session("session_delta"))

        # Check key and ciphertext erasure
        self.assertIsNone(session._key)
        self.assertIsNone(session._cipher)
        self.assertIsNone(session._encrypted_profile)
        self.assertIsNone(session._encrypted_last_good_render)
        self.assertIsNone(session._encrypted_face_mask)

    def test_face_mask_encryption_and_caching(self):
        """Verify face & hands mask is securely cached encrypted and retrieved accurately."""
        import numpy as np
        session = self.manager.get_or_create_session("session_mask_test")
        orig_mask = np.random.uniform(0.0, 1.0, size=(120, 90)).astype(np.float32)
        session.set_face_mask(orig_mask)

        # Confirm stored as ciphertext
        self.assertIsInstance(session._encrypted_face_mask, bytes)

        # Retrieve and verify exact numerical equivalence
        retrieved_mask = session.get_face_mask()
        self.assertIsNotNone(retrieved_mask)
        self.assertEqual(retrieved_mask.shape, (120, 90))
        np.testing.assert_allclose(retrieved_mask, orig_mask, rtol=1e-5)

    def test_outfit_scoped_caching_and_invalidation(self):
        """
        Verify size renders and canonical renders are scoped to dress_hash,
        and invalidating outfit cache removes garment renders while strictly preserving body_profile.
        """
        session = self.manager.get_or_create_session("session_outfit_test")
        session.set_body_profile(self.sample_profile)

        import numpy as np
        sample_mask = np.ones((50, 50), dtype=np.float32)
        session.set_face_mask(sample_mask)

        # Set renders for Garment A
        hash_a = "dress_hash_aaa"
        session.set_last_good_render("render_a_standard", dress_hash=hash_a)
        session.set_canonical_render("render_a_canonical", dress_hash=hash_a)
        session.set_size_render("M", "render_a_size_m", dress_hash=hash_a)
        session.set_size_render("L", "render_a_size_l", dress_hash=hash_a)

        # Confirm Garment A cache hits
        self.assertEqual(session.get_canonical_render(dress_hash=hash_a), "render_a_canonical")
        self.assertEqual(session.get_size_render("M", dress_hash=hash_a), "render_a_size_m")
        self.assertEqual(session.get_size_render("L", dress_hash=hash_a), "render_a_size_l")

        # Confirm Garment B does NOT hit Garment A's cache
        hash_b = "dress_hash_bbb"
        self.assertIsNone(session.get_canonical_render(dress_hash=hash_b))
        self.assertIsNone(session.get_size_render("M", dress_hash=hash_b))

        # Switch outfit to Garment B and invalidate
        session.invalidate_outfit_cache(new_dress_hash=hash_b)

        # Renders for old garment are cleared
        self.assertIsNone(session.get_canonical_render(dress_hash=hash_a))
        self.assertIsNone(session.get_size_render("M", dress_hash=hash_a))
        self.assertIsNone(session.get_last_good_render(dress_hash=hash_a))

        # CRITICAL: Biometric body profile and face mask MUST remain intact across garment switches!
        preserved_profile = session.get_body_profile()
        self.assertIsNotNone(preserved_profile)
        self.assertEqual(preserved_profile.body_type, "Athletic / V-Shape")
        self.assertAlmostEqual(preserved_profile.shoulder_width_ratio, 0.28)

        preserved_mask = session.get_face_mask()
        self.assertIsNotNone(preserved_mask)
        np.testing.assert_allclose(preserved_mask, sample_mask)


if __name__ == "__main__":
    unittest.main()
