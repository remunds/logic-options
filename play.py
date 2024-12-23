from logic_options.envs.renderer import Renderer

if __name__ == "__main__":
     renderer = Renderer(env_name="ALE/Seaquest-v5",
     #renderer = Renderer(env_name="MeetingRoom",
                        #agent_name="neural_pretrained_maggi_trainable_1",
                        #agent_name="neural_hierarchy_polterm_weak_very_low_ent",
                        agent_name="neural_flat_surface_4",
                    #     agent_name="debug",
                        #agent_name="neural_hierarchy_polterm_none_1",
                        #agent_name="elevator_option_15",
                        #agent_name="target_option_10",
                        fps=15,
                        deterministic=True,
                        shadow_mode=False,
                        wait_for_input=False,
                        render_oc_overlay=False,
                        render_predicate_probs=False)
     renderer.run()
