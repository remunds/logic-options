from logic_options.envs.renderer import Renderer

if __name__ == "__main__":
    # renderer = Renderer(env_name="ALE/Kangaroo-v5",
    #                     agent_name="kangaroo_ill_def_rerun_8",
    renderer = Renderer(env_name="MeetingRoom",
                        agent_name="elevator_option_12",
                        fps=15,
                        deterministic=True,
                        shadow_mode=True,
                        wait_for_input=True,
                        render_predicate_probs=False)
    renderer.run()
