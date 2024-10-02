import rpy2.rlike.container as rlc


def rnamedlist(d: dict):
  return rlc.TaggedList([e if not isinstance(e, dict) else rnamedlist(e) for e in d.values()], [k for k in d.keys()])