from envs.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(env_name="ALE/Seaquest-v5",
                        agent_name="debug",
                        fps=15,
                        shadow_mode=True,
                        deterministic=True,
                        wait_for_input=False)
    renderer.run()
