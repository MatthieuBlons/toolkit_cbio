class HookerAttnMIL:
    """
    Manage the hooks for the MIL algorithm.
    Works with a MIL instantiated with an MHMC model.
    """

    def __init__(self, network, num_heads):
        self.num_heads = num_heads
        self.place_hooks(network)

        # Possibly filled attribute
        self.tiles_transf = None
        self.tiles_weights = None
        self.head_average = None
        self.reprewsi = None
        self.scores = None

    def _get_instance_transf(self):
        def hook_instance_transf(m, i, o):
            """
            hooks the output of the attention heads
            """
            tiles_transf = i[0]
            self.tiles_transf = tiles_transf.detach().cpu().numpy()

        return hook_instance_transf

    def _get_attention_hook(self):
        def hook_attention(m, i, o):
            """
            hooks the output of the attention heads
            """
            tiles_weights = i[0]
            tiles_weights = tiles_weights.view(-1, self.num_heads)
            self.tiles_weights = tiles_weights.detach().cpu().numpy()

        return hook_attention

    def _get_outputs_hook(self):
        def hook_output(m, i, o):
            """Hooks the outputs of the classifier, before the
            logsoftmax layer(expected to saturate).
            """
            repre = i[0]
            self.scores = repre.detach().cpu().numpy()

        return hook_output

    def _get_average_hook(self):
        def hook_head_average(m, i, o):
            """Hooks the entry of the MLP classifier
            i.e the different average tiles
            """
            repre = i[0].view(self.num_heads, -1)
            self.head_average = repre.detach().cpu().numpy()

        return hook_head_average

    def _get_wsi_representation_hook(self):
        def hook_reprewsi(m, i, o):
            """
            Hooks the antepenultiem layer of the classifier.
            """
            repre = i[0]
            self.reprewsi = repre.squeeze().detach().cpu().numpy()

        return hook_reprewsi

    def place_hooks(self, net):
        for name, layer in net.named_children():
            if list(layer.children()):
                self.place_hooks(layer)

            if name == "instance_transf":
                hook_layer = list(layer.children())[0]  # last layer is softmax
                hook_layer.register_forward_hook(self._get_instance_transf())

            if name == "attention":
                hook_layer = list(layer.children())[-1]  # last layer is softmax
                hook_layer.register_forward_hook(self._get_attention_hook())

            if name == "classifier":
                layer_avg = list(layer.children())[0]
                layer_avg.register_forward_hook(self._get_average_hook())

                layer_reprewsi = list(layer.children())[-2]
                layer_reprewsi.register_forward_hook(
                    self._get_wsi_representation_hook()
                )

                layer_scores = list(layer.children())[-1]
                layer_scores.register_forward_hook(self._get_outputs_hook())
