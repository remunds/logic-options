from envs.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(env_name="ALE/Seaquest-v5",
                        agent_name="logic/4",
                        fps=15,
                        shadow_mode=False,
                        deterministic=False,
                        wait_for_input=False)
    renderer.run()
