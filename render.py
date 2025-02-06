from uno_env import UNOGame, Card, CardColor, CardType, UNORenderer


class ASCIIUnoRenderer(UNORenderer):
    COLOR_CODES = {
        CardColor.RED: "\033[41m",  # 红底
        CardColor.YELLOW: "\033[43m",  # 黄底
        CardColor.BLUE: "\033[44m",  # 蓝底
        CardColor.GREEN: "\033[42m",  # 绿底
        CardColor.BLACK: "\033[40m",  # 黑底
    }
    RESET_CODE = "\033[0m"
    MAX_DISCARD_DISPLAY = 10

    def _card_repr(self, card: Card) -> str:
        """单个卡牌的字符表示"""
        color_code = self.COLOR_CODES[card.color]
        text_color = "\033[37m"  # 白色文字

        # 处理特殊牌
        symbol = card.value
        if card.type == CardType.SKIP:
            symbol = "S"
        elif card.type == CardType.REVERSE:
            symbol = "R"
        elif card.type == CardType.DRAW_TWO:
            symbol = "+2"
        elif card.type == CardType.WILD:
            symbol = "W"
        elif card.type == CardType.WILD_DRAW_FOUR:
            symbol = "W4"

        return f"{color_code}{text_color}{symbol:^4}{self.RESET_CODE}"

    def render(self, env: UNOGame, show_all=True):
        current_color = env.current_color
        """渲染游戏界面"""
        print("\033c")  # 清屏

        # 顶部信息
        print(f"=== UNO Game (Players: {env.num_players}) ===")
        print(f"Direction: {'→' if env.direction == 1 else '←'}")
        print(
            f"Current Color: {self.COLOR_CODES[current_color]}{current_color.value}{self.RESET_CODE}"
        )

        # 弃牌堆显示
        if env.discard_pile:
            shown_cards = [
                self._card_repr(card)
                for card in env.discard_pile[-self.MAX_DISCARD_DISPLAY :][::-1]
            ]
            print("\nDiscard Pile:", " ".join(shown_cards), end="")
            if len(env.discard_pile) > self.MAX_DISCARD_DISPLAY:
                print(
                    f" (+{len(env.discard_pile) - self.MAX_DISCARD_DISPLAY} more)",
                    end="",
                )
            print("")

        else:
            print("\nDiscard Pile: [Empty]")

        # 玩家信息
        print("\nPlayers:")
        for i, player in enumerate(env.players):
            status = ""
            if i == env.current_player:
                status = " ← CURRENT"
            if show_all or i == env.current_player:
                cards = " ".join([self._card_repr(c) for c in player.hand])
                print(f"Player {i+1} ({len(player.hand)} cards){status}:")
                if i == env.current_player or show_all:
                    print(f"  {cards}")
                else:
                    print("  [Cards hidden]")
            else:
                print(f"Player {i+1}: {len(player.hand)} cards{status}")

        # 牌堆信息
        print(f"\nDeck remaining: {len(env.deck)} cards")
        print("=" * 40)

    def user_input(self, env: UNOGame, valid_actions: list) -> int:
        """用户输入"""
        print("Available actions:")
        for i, act in enumerate(valid_actions):
            if act < 108:
                card = env._action_to_card(act)
                print(f"{i}: {self._card_repr(card)}", end=" ")
            else:
                print(f"{i}: Draw Card")
        while True:
            try:
                choice = int(input("\nSelect action: "))
                if choice < 0 or choice >= len(valid_actions):
                    raise ValueError()
                break
            except ValueError:
                print("Invalid choice. Try again.")

        return choice
