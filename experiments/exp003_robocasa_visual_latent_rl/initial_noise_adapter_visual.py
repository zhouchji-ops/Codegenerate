"""Isolated initial-noise adapter; no disk edits to official model or global monkeypatch.
Original forward is compiled with exactly one RNG expression replacement.
Caller must retain source license and validate full-model numerical parity.
"""
import ast, inspect, textwrap, types, hashlib
import torch

def choose_noise(mask, initial_noise=None):
    if initial_noise is None:
        return torch.randn_like(mask)
    if initial_noise.shape != mask.shape:
        raise ValueError("noise shape must exactly match action_mask")
    if initial_noise.dtype != mask.dtype or initial_noise.device != mask.device:
        raise ValueError("noise dtype/device must match action_mask")
    if not torch.isfinite(initial_noise).all():
        raise ValueError("initial_noise must be finite")
    return initial_noise.clone()

def adapt_forward(original, noise_selector=choose_noise):
    original = inspect.unwrap(original)
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    fn = tree.body[0]
    if "initial_noise" in inspect.signature(original).parameters:
        raise ValueError("already adapted")
    hits = 0
    class Replace(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal hits
            if (len(node.targets)==1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id=="x"
                and ast.unparse(node.value)=="torch.randn_like(action_mask)"):
                hits += 1
                node.value=ast.parse("_choose_noise(action_mask, initial_noise)",mode="eval").body
            return self.generic_visit(node)
    tree=Replace().visit(tree)
    if hits != 1:
        raise ValueError("expected exactly one recognized initial-noise expression")
    fn.args.kwonlyargs.append(ast.arg(arg="initial_noise"))
    fn.args.kw_defaults.append(ast.Constant(None))
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__)
    namespace["_choose_noise"]=noise_selector
    exec(compile(tree,"<isolated_xr1_initial_noise>","exec"),namespace)
    return namespace[fn.name], hashlib.sha256(source.encode()).hexdigest()

def bind_instance(model, noise_selector=choose_noise):
    original=model.forward.__func__
    adapted, source_sha256=adapt_forward(original, noise_selector=noise_selector)
    model.forward=types.MethodType(adapted,model)
    return {"original_forward_sha256":source_sha256,"replacement_count":1,
            "scope":"this model instance only","full_model_parity":"not yet tested"}
