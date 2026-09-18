#!/usr/bin/env python3

"""
Unit tests for the SHRINCS reference implementation.

Run them from this directory with `python3 test.py`, or from the repository
root with `python3 -m impl.test`.
"""

import functools
import unittest
from contextlib import contextmanager
from typing import get_args, get_type_hints

if __package__:
  from . import shrincs
else:
  import shrincs


#  Test fixtures
#
#  Every value below is fixed rather than sampled, so that a failure reproduces
#  exactly. Nothing under test depends on a seed being unpredictable: the
#  implementation is checked for what it derives from a seed, never for how the
#  seed was drawn. The shared fixtures below are deliberately not named `test_*`,
#  so that a third-party runner does not mistake them for test cases.

TEST_SEED    = bytes(range(48))
TEST_MESSAGE = b"foobar!"
TEST_CTX     = b"SHRINCS unit tests"

#  The FXMSS structure the stateful tests share: a balanced tree of depth 4, the
#  smallest shape that exercises every stateful ADRS type, including the
#  interior merkle nodes of an authentication path.
TEST_SF_STRUCTURE = bytes([shrincs.FXMSS_SHAPE_BALANCED, 4])


@functools.lru_cache(maxsize = None)
def shared_keypair(sf_structure: bytes) -> tuple[bytes, bytes]:
  """
  The SHRINCS keypair the tests share for a given FXMSS tree structure.

  Key generation walks the entire stateless root tree, which is by far the most
  expensive thing this suite does, so it is cached to once per structure.
  """
  return shrincs.shrincs_keygen(TEST_SEED, sf_structure)

@functools.lru_cache(maxsize = None)
def shared_stateless_signature(sf_structure: bytes, message: bytes, ctx: bytes) -> bytes:
  """
  A stateless SHRINCS signature the tests share. Stateless signing traverses
  every FORS tree and every hypertree layer, so it too is cached.
  """
  seckey, _ = shared_keypair(sf_structure)
  return shrincs.shrincs_sign(message, ctx, seckey, None, None)


def find_metadata(annotation, metadata_type) -> list:
  """
  Collects every piece of `metadata_type` metadata attached to a type
  annotation, at any depth.
  """
  if isinstance(annotation, metadata_type):
    return [annotation]
  return [
    metadata
    for argument in get_args(annotation)
    for metadata in find_metadata(argument, metadata_type)
  ]


#  ADRS type tags, read off the implementation by name so that a tag added later
#  is held to the same rules without this file being touched.

def adrs_types(prefix: str) -> set[int]:
  return {
    value for name, value in vars(shrincs).items()
    if name.startswith(prefix) and isinstance(value, int)
  }

SL_ADRS_TYPES = adrs_types('SL_')
SF_ADRS_TYPES = adrs_types('SF_')

#  Every tweaked hash function, and the position of its `ADRS` argument.
ADRS_ARGUMENT = {
  'T_sl': 1, 'T_sf': 1, 'T_k': 1, 'F': 1, 'H': 1, 'H_grind': 1, 'PRF': 2,
}

@contextmanager
def recorded_adrs_types():
  """
  Yields the set of ADRS type tags used by every tweaked hash computed inside
  the block.

  The implementation reaches its tweaked hash functions through module globals,
  so replacing them here observes every call any algorithm makes, however deep
  in the recursion.
  """
  observed = set()
  originals = {name: getattr(shrincs, name) for name in ADRS_ARGUMENT}

  def recording(function, position):
    @functools.wraps(function)
    def recorder(*args, **kwargs):
      ADRS = kwargs['ADRS'] if 'ADRS' in kwargs else args[position]
      observed.add(ADRS[9])
      return function(*args, **kwargs)
    return recorder

  for name, position in ADRS_ARGUMENT.items():
    setattr(shrincs, name, recording(originals[name], position))
  try:
    yield observed
  finally:
    for name, function in originals.items():
      setattr(shrincs, name, function)


class ValueTypeTest(unittest.TestCase):
  """
  The value type annotations are part of the specification: they state the range
  an implementation must accommodate. These tests hold the annotations to what
  the surrounding prose claims.
  """

  def test_winternitz_chain_iterators_count_in_32_bit_words(self):
    for chain_iterator in (shrincs.wots_tw_chain_iter, shrincs.wots_c_chain_iter):
      with self.subTest(chain_iterator = chain_iterator.__name__):
        annotations = get_type_hints(chain_iterator, include_extras = True)
        self.assertEqual(find_metadata(annotations['start'], shrincs.UINT)[0].bits, 32)
        self.assertEqual(find_metadata(annotations['steps'], shrincs.UINT)[0].bits, 32)

  def test_sign_returns_a_shrincs_signature_not_a_bare_sphincs_one(self):
    return_lengths = find_metadata(
      get_type_hints(shrincs.shrincs_sign, include_extras = True)['return'], shrincs.LEN
    )
    self.assertTrue(
      any(length.size == shrincs.SHRINCS_SL_SIGNATURE_SIZE for length in return_lengths)
    )
    self.assertFalse(
      any(length.size == shrincs.SPHX_SIGNATURE_SIZE for length in return_lengths)
    )


class SignVerifyTest(unittest.TestCase):
  """
  End-to-end signing and verification over each FXMSS tree shape.
  """

  #  Each structure, paired with the number of stateful signatures it can issue.
  STRUCTURES = [
    (bytes([shrincs.FXMSS_SHAPE_BALANCED, 4]), 16),
    (bytes([shrincs.FXMSS_SHAPE_UNBALANCED, 16]), 17),
  ]

  def test_every_stateful_signature_slot_verifies(self):
    for sf_structure, signature_count in self.STRUCTURES:
      seckey, pubkey = shared_keypair(sf_structure)
      for state_ctr in range(signature_count):
        with self.subTest(sf_structure = sf_structure.hex(), state_ctr = state_ctr):
          signature = shrincs.shrincs_sign(TEST_MESSAGE, TEST_CTX, seckey, state_ctr, None)
          self.assertIsNotNone(signature)
          self.assertTrue(shrincs.shrincs_verify(TEST_MESSAGE, signature, TEST_CTX, pubkey))

  def test_the_stateless_signature_verifies(self):
    _, pubkey = shared_keypair(TEST_SF_STRUCTURE)
    signature = shared_stateless_signature(TEST_SF_STRUCTURE, TEST_MESSAGE, TEST_CTX)
    self.assertEqual(len(signature), shrincs.SHRINCS_SL_SIGNATURE_SIZE)
    self.assertTrue(shrincs.shrincs_verify(TEST_MESSAGE, signature, TEST_CTX, pubkey))


class DomainSeparationTest(unittest.TestCase):
  """
  The stateful and stateless components share one set of seeds, one 22-byte ADRS
  layout and one underlying hash function, so nothing but deliberate separation
  keeps them apart. Each test below pins down one of the mechanisms that does
  the keeping: the ADRS type tag, the padding of a key or a preimage, or the
  indicator byte a signature carries.
  """

  #  Arbitrary but fixed inputs for the hash-level tests. Their values do not
  #  matter; what matters is that both functions under test are handed the same
  #  ones, so that only the separation mechanism can tell the outputs apart.
  R       = bytes(range(0,  16))
  PK_SEED = bytes(range(16, 32))
  ROOT    = bytes(range(32, 48))
  SK_PRF  = bytes(range(48, 64))
  ADRS    = bytearray(range(22))

  def test_adrs_type_tags_are_disjoint(self):
    """
    Both paths address a node by one byte of layer or height followed by eight
    bytes of tree or node index, and those ranges overlap: FXMSS height 2, node
    5 occupies the same nine leading bytes as hypertree layer 2, tree 5. The
    type tag at `ADRS[9]` is the only field that separates them, so no value may
    serve both paths.
    """
    self.assertTrue(SL_ADRS_TYPES)
    self.assertTrue(SF_ADRS_TYPES)

    #  The tags are written into a single byte of the address.
    for tag in SL_ADRS_TYPES | SF_ADRS_TYPES:
      with self.subTest(tag = tag):
        self.assertIn(tag, range(256))

    #  Stateless tags are assigned from zero and stateful ones from 16, so the
    #  two paths cannot meet even as either set grows.
    self.assertLess(max(SL_ADRS_TYPES), 16)
    self.assertGreaterEqual(min(SF_ADRS_TYPES), 16)
    self.assertFalse(SL_ADRS_TYPES & SF_ADRS_TYPES)

  def test_stateful_signing_and_verification_stay_stateful(self):
    """
    Every hash a stateful signature commits to must be one no stateless
    algorithm can reach, so every tweaked hash on the stateful path must carry a
    stateful tag. Signing uses all of them; verification uses all but the PRF
    tag, holding no secret to derive a preimage from.
    """
    seckey, pubkey = shared_keypair(TEST_SF_STRUCTURE)

    with recorded_adrs_types() as observed:
      signature = shrincs.shrincs_sign(TEST_MESSAGE, TEST_CTX, seckey, 0, None)
    self.assertIsNotNone(signature)
    self.assertEqual(observed, SF_ADRS_TYPES)

    with recorded_adrs_types() as observed:
      self.assertTrue(shrincs.shrincs_verify(TEST_MESSAGE, signature, TEST_CTX, pubkey))
    self.assertEqual(observed, SF_ADRS_TYPES - {shrincs.SF_WOTS_C_PRF})

  def test_stateless_signing_and_verification_stay_stateless(self):
    """
    The mirror of the stateful case: signing reaches every stateless tag, and
    verification reaches every one that is not used to derive a secret.
    """
    seckey, pubkey = shared_keypair(TEST_SF_STRUCTURE)

    with recorded_adrs_types() as observed:
      signature = shrincs.shrincs_sign(TEST_MESSAGE, TEST_CTX, seckey, None, None)
    self.assertIsNotNone(signature)
    self.assertEqual(observed, SL_ADRS_TYPES)

    with recorded_adrs_types() as observed:
      self.assertTrue(shrincs.shrincs_verify(TEST_MESSAGE, signature, TEST_CTX, pubkey))
    self.assertEqual(
      observed, SL_ADRS_TYPES - {shrincs.SL_WOTS_TW_PRF, shrincs.SL_FORS_PRF}
    )

  def test_root_computations_stay_in_their_own_domain(self):
    """
    Key generation derives both roots from one `sk_seed` and one `pk_seed`, so
    the two tree walks must not meet either. Neither walk grinds or touches
    FORS, so those tags are the ones absent.
    """
    sk_seed, pk_seed = TEST_SEED[0:16], TEST_SEED[32:48]

    ADRS = bytearray(22)
    ADRS[0] = shrincs.SPHX_LAYER_COUNT - 1
    with recorded_adrs_types() as observed:
      shrincs.xmss_node(sk_seed, 0, shrincs.SPHX_XMSS_HEIGHT, pk_seed, ADRS)
    self.assertEqual(
      observed,
      SL_ADRS_TYPES - {shrincs.SL_FORS_TREE, shrincs.SL_FORS_ROOTS, shrincs.SL_FORS_PRF},
    )

    tree_balanced = TEST_SF_STRUCTURE[0] == shrincs.FXMSS_SHAPE_BALANCED
    with recorded_adrs_types() as observed:
      shrincs.fxmss_node(
        sk_seed, 0, shrincs.FXMSS_HEIGHT, pk_seed,
        tree_balanced, TEST_SF_STRUCTURE[1], bytearray(22),
      )
    self.assertEqual(observed, SF_ADRS_TYPES - {shrincs.SF_WOTS_C_GRIND})

  def test_winternitz_chains_are_domain_separated(self):
    """
    The two Winternitz chains iterate the same `F` over the same address layout,
    so the tag each iterator writes into `ADRS[9]` is all that keeps a WOTS+C
    chain and a WOTS-TW chain at the same tree position apart.
    """
    node = bytes(range(16))
    stateless = shrincs.wots_tw_chain_iter(node, 0, 1, self.PK_SEED, bytearray(22))
    stateful  = shrincs.wots_c_chain_iter(node, 0, 1, self.PK_SEED, bytearray(22))
    self.assertNotEqual(stateless, stateful)

  def test_winternitz_public_keys_are_domain_separated(self):
    """
    A WOTS-TW and a WOTS+C keypair generated from the same seeds at the same
    address must not share a public key. Each step between the seed and the
    public key is tagged for one path only: the secret preimages, the chain
    iteration, and the compression of the chain tips.
    """
    sk_seed = TEST_SEED[0:16]
    stateless = shrincs.wots_tw_pubkey_gen(sk_seed, self.PK_SEED, bytearray(22))
    stateful  = shrincs.wots_c_pubkey_gen(sk_seed, self.PK_SEED, bytearray(22))
    self.assertNotEqual(stateless, stateful)

  def test_message_digests_are_domain_separated(self):
    """
    `H_msg_sl` and `H_msg_sf` both hash `R || pk_seed || <inner digest> ||
    <tail>`. Lining their inner digests up exactly, by prepending the nine ADRS
    bytes to the stateless message, leaves only the tail to separate them: four
    zero bytes against those same nine ADRS bytes. The two outer preimages
    therefore differ in length whatever the inputs, so no choice of message can
    bring the two digests together.
    """
    stateful = shrincs.H_msg_sf(
      self.R, self.PK_SEED, self.ROOT, self.ADRS, TEST_MESSAGE
    )
    stateless = shrincs.H_msg_sl(
      self.R, self.PK_SEED, self.ROOT, bytes(self.ADRS[:9]) + TEST_MESSAGE
    )
    self.assertNotEqual(stateful, stateless)

  def test_message_randomizers_are_domain_separated(self):
    """
    `PRF_msg_sl` and `PRF_msg_sf` are both HMAC-SHA256 under the same 16-byte
    `sk_prf`. Handing them the same HMAC message, by taking `opt_rand` as the
    public seed and prepending the nine ADRS bytes, leaves only the key block to
    separate them: HMAC pads the stateless key to 64 bytes with zeros, while the
    stateful path pads the same secret with 0xFF.
    """
    stateful = shrincs.PRF_msg_sf(
      self.SK_PRF, self.PK_SEED, self.ADRS, TEST_MESSAGE
    )
    stateless = shrincs.PRF_msg_sl(
      self.SK_PRF, self.PK_SEED, bytes(self.ADRS[:9]) + TEST_MESSAGE
    )
    self.assertNotEqual(stateful, stateless)

  def test_the_indicator_byte_separates_the_two_signature_forms(self):
    """
    The indicator byte tells the verifier which component produced a signature,
    and so which root it must recompute. Retagging a signature as the other
    component's must not produce something a verifier accepts.
    """
    seckey, pubkey = shared_keypair(TEST_SF_STRUCTURE)
    stateful  = shrincs.shrincs_sign(TEST_MESSAGE, TEST_CTX, seckey, 0, None)
    stateless = shared_stateless_signature(TEST_SF_STRUCTURE, TEST_MESSAGE, TEST_CTX)

    self.assertEqual(stateless[0], shrincs.FXMSS_HEIGHT)
    self.assertLess(stateful[0], shrincs.FXMSS_HEIGHT)

    retagged_as_stateful = bytes([stateful[0]]) + stateless[1:]
    self.assertFalse(
      shrincs.shrincs_verify(TEST_MESSAGE, retagged_as_stateful, TEST_CTX, pubkey)
    )

    retagged_as_stateless = bytes([shrincs.FXMSS_HEIGHT]) + stateful[1:]
    self.assertFalse(
      shrincs.shrincs_verify(TEST_MESSAGE, retagged_as_stateless, TEST_CTX, pubkey)
    )

  def test_a_stateless_signature_is_bound_to_the_stateful_root(self):
    """
    A stateless signature covers `sf_root || message`, which ties it to the
    stateful half of the very keypair it was made under. Moving it to a key that
    differs only in `sf_root` must not carry the signature with it.
    """
    _, pubkey = shared_keypair(TEST_SF_STRUCTURE)
    signature = shared_stateless_signature(TEST_SF_STRUCTURE, TEST_MESSAGE, TEST_CTX)
    self.assertTrue(shrincs.shrincs_verify(TEST_MESSAGE, signature, TEST_CTX, pubkey))

    foreign = pubkey[0:32] + bytes([pubkey[32] ^ 1]) + pubkey[33:48]
    self.assertFalse(shrincs.shrincs_verify(TEST_MESSAGE, signature, TEST_CTX, foreign))

  def test_a_stateful_signature_is_bound_to_the_stateless_root(self):
    """
    The mirror of the above: a stateful signature covers `sl_root || message`,
    so a key that differs only in `sl_root` must not verify it.
    """
    seckey, pubkey = shared_keypair(TEST_SF_STRUCTURE)
    signature = shrincs.shrincs_sign(TEST_MESSAGE, TEST_CTX, seckey, 0, None)
    self.assertTrue(shrincs.shrincs_verify(TEST_MESSAGE, signature, TEST_CTX, pubkey))

    foreign = pubkey[0:16] + bytes([pubkey[16] ^ 1]) + pubkey[17:48]
    self.assertFalse(shrincs.shrincs_verify(TEST_MESSAGE, signature, TEST_CTX, foreign))

  def test_keys_differing_only_in_tree_structure_do_not_share_signatures(self):
    """
    Two keys generated from one seed under different FXMSS structures agree on
    everything the stateless path derives, and differ only in `sf_root`. Because
    each component is bound to the other's root, neither key's signatures verify
    under the other, on either path. This is the same binding the two tests
    above isolate, reached here through keys a signer could really hold rather
    than through a spliced public key.
    """
    balanced   = TEST_SF_STRUCTURE
    unbalanced = bytes([shrincs.FXMSS_SHAPE_UNBALANCED, 16])

    balanced_seckey, balanced_pubkey     = shared_keypair(balanced)
    unbalanced_seckey, unbalanced_pubkey = shared_keypair(unbalanced)

    #  The two keys share their public seed and stateless root, and part ways at
    #  the stateful root.
    self.assertEqual(balanced_pubkey[0:32], unbalanced_pubkey[0:32])
    self.assertNotEqual(balanced_pubkey[32:48], unbalanced_pubkey[32:48])

    for signing_key, foreign_pubkey in (
      (balanced_seckey, unbalanced_pubkey),
      (unbalanced_seckey, balanced_pubkey),
    ):
      with self.subTest(sf_structure = signing_key[64:66].hex()):
        stateful = shrincs.shrincs_sign(TEST_MESSAGE, TEST_CTX, signing_key, 0, None)
        self.assertFalse(
          shrincs.shrincs_verify(TEST_MESSAGE, stateful, TEST_CTX, foreign_pubkey)
        )

    stateless = shared_stateless_signature(balanced, TEST_MESSAGE, TEST_CTX)
    self.assertFalse(
      shrincs.shrincs_verify(TEST_MESSAGE, stateless, TEST_CTX, unbalanced_pubkey)
    )


if __name__ == "__main__":
  unittest.main()
