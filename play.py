from envs.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(env_name="ALE/Seaquest-v5",
                        agent_name="neural/flat/2-many-steps",
                        fps=15,
                        deterministic=False,
                        shadow_mode=False,
                        wait_for_input=False,
                        render_predicate_probs=False)
    renderer.run()
