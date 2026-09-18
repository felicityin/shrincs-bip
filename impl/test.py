from random import randbytes
from typing import get_args, get_type_hints
from shrincs import shrincs_sign, shrincs_keygen, shrincs_verify
from shrincs import FXMSS_SHAPE_UNBALANCED, FXMSS_SHAPE_BALANCED, FXMSS_HEIGHT
from shrincs import LEN, UINT, SHRINCS_SL_SIGNATURE_SIZE, SPHX_SIGNATURE_SIZE
from shrincs import wots_c_chain_iter, wots_tw_chain_iter
from shrincs import hash_shrincs_sign, hash_shrincs_verify
from shrincs import shrincs_sign_internal, slh_dsa_sign_internal
from shrincs import MSG_DOMAIN_PURE, MSG_DOMAIN_PREHASH, PH_OID_SHA256, sha256


def find_metadata(annotation, metadata_type):
  if isinstance(annotation, metadata_type):
    return [annotation]
  return [
    metadata
    for argument in get_args(annotation)
    for metadata in find_metadata(argument, metadata_type)
  ]

if __name__ == "__main__":
  for chain_iterator in (wots_tw_chain_iter, wots_c_chain_iter):
    annotations = get_type_hints(chain_iterator, include_extras=True)
    assert find_metadata(annotations['start'], UINT)[0].bits == 32
    assert find_metadata(annotations['steps'], UINT)[0].bits == 32

  return_lengths = find_metadata(
    get_type_hints(shrincs_sign, include_extras=True)['return'], LEN
  )
  assert any(length.size == SHRINCS_SL_SIGNATURE_SIZE for length in return_lengths)
  assert not any(length.size == SPHX_SIGNATURE_SIZE for length in return_lengths)

  structures = [
    (bytes([FXMSS_SHAPE_BALANCED, 4]), 16),
    (bytes([FXMSS_SHAPE_UNBALANCED, 16]), 17)
  ]
  msg = b"foobar!"
  #  Longer than any hash input in the scheme, which pre-hash signing does not mind.
  long_msg = randbytes(1 << 20)

  for (i, (sf_structure, stateful_signature_count)) in enumerate(structures):
    sk, pk = shrincs_keygen(randbytes(48), sf_structure)

    for j in range(stateful_signature_count):
      #  Alternate the two signing forms across the key's stateful slots.
      pure = j % 2 == 0
      sign = shrincs_sign if pure else hash_shrincs_sign
      verify, other_verify = (shrincs_verify, hash_shrincs_verify) if pure \
                        else (hash_shrincs_verify, shrincs_verify)
      signed = msg if pure else long_msg

      sig = sign(signed, b"", sk, j, None)
      assert verify(signed, sig, b"", pk)
      assert not verify(signed + b"!", sig, b"", pk)
      assert not verify(signed, sig, b"other", pk)
      #  The domain separator keeps the two forms apart.
      assert not other_verify(signed, sig, b"", pk)

      if not pure:
        #  A signer holding only the digest reaches the same signature.
        payload = PH_OID_SHA256 + sha256(signed)
        assert sig == shrincs_sign_internal(MSG_DOMAIN_PREHASH, b"", payload, sk, j, None)
    print(f'verified all stateful signatures for structure {sf_structure.hex()}')

    if i == len(structures) - 1:
      sig = shrincs_sign(msg, b"", sk, None, None)
      assert shrincs_verify(msg, sig, b"", pk)
      assert not hash_shrincs_verify(msg, sig, b"", pk)
      assert len(sig) == SHRINCS_SL_SIGNATURE_SIZE
      print(f'verified stateless signature')

      #  The stateless path stays reachable through the FIPS-205 interface: pure
      #  signing binds `sf_root || message` just as `slh_sign` would.
      sk_seed, sk_prf, pk_seed, sl_root, sf_root = sk[0:16], sk[16:32], sk[32:48], sk[48:64], sk[66:82]
      bound_message = bytes([MSG_DOMAIN_PURE, 0]) + sf_root + msg
      assert sig[1:] == slh_dsa_sign_internal(bound_message, sk_seed, sk_prf, pk_seed, sl_root, None)
      print(f'verified stateless signature against the SLH-DSA internal interface')
