from torchviz import make_dot

def do_torchviz_plots(loss, actor_net, critic_net, target_value, value_net, update_name):
    make_dot(loss, params=dict(**dict(actor_net.named_parameters(prefix='actor')), 
                                **dict(critic_net.named_parameters(prefix='critic')), 
                                **dict(target_value.named_parameters(prefix='target_value')), 
                                **dict(value_net.named_parameters(prefix='value'))), 
                                show_attrs=True, show_saved=True).render('torchviz_plots/' + update_name, format="svg")
