from envs.renderer import Renderer

if __name__ == "__main__":
    # renderer = Renderer(env_name="ALE/Seaquest-v5",
    #                     agent_name="neural/flat/2-many-steps",
    renderer = Renderer(env_name="ALE/Kangaroo-v5",
                        agent_name="kangaroo_ill_def_rerun_8",
                        fps=15,
                        deterministic=False,
                        shadow_mode=False,
                        wait_for_input=False,
                        render_predicate_probs=False)
    renderer.run()
