import random
from enum import Enum
from typing import List, Tuple, Optional
from gym import Env, spaces
from abc import ABC, abstractmethod

# --------------------------
# 基本数据模型定义
# --------------------------


class CardType(Enum):
    NUMBER = "number"
    SKIP = "skip"
    REVERSE = "reverse"
    DRAW_TWO = "draw_two"
    WILD = "wild"
    WILD_DRAW_FOUR = "wild_draw_four"


class CardColor(Enum):
    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"
    GREEN = "green"
    BLACK = "black"


class Card:
    def __init__(
        self, color: CardColor, value: str, card_type: CardType, unique_id: int
    ):
        self.color = color
        self.value = value
        self.type = card_type
        self.unique_id = unique_id

    def __repr__(self):
        return f"{self.color.value}_{self.value}"

    def __eq__(self, other):
        return self.unique_id == other.unique_id


# --------------------------
# 异常类
# --------------------------


class NoMoreCardsError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


# --------------------------
# 游戏核心逻辑
# --------------------------


class UNOGame(Env):
    def __init__(self, num_players: int = 2, renderer: "UNORenderer" = None):
        super().__init__()

        self.all_cards = self._create_all_card()

        # 动作空间：0-107代表108种牌，108代表摸牌
        self.action_space = spaces.Discrete(109)

        # 状态空间（简化版）
        self.observation_space = spaces.Dict(
            {
                "hand": spaces.MultiBinary(108),
                "current_color": spaces.Discrete(5),
                "direction": spaces.Discrete(2),
                "player_turn": spaces.Discrete(num_players),
                "discard_pile": spaces.MultiBinary(108),
                "deck_size": spaces.Discrete(108),
            }
        )

        self.num_players = num_players
        self.players: List[Player] = []
        self.deck: List[Card] = []
        self.discard_pile: List[Card] = []
        self.current_player_idx = 0
        self.direction = 1
        self.current_color: Optional[CardColor] = None
        self.renderer = renderer

    def _create_all_card(self):
        """创建所有的卡牌"""
        cards = []

        def add_card(color, value, card_type):
            cards.append(Card(color, value, card_type, len(cards)))

        # 数字牌
        for color in [c for c in CardColor if c != CardColor.BLACK]:
            add_card(color, "0", CardType.NUMBER)
            for num in range(1, 10):
                add_card(color, str(num), CardType.NUMBER)
                add_card(color, str(num), CardType.NUMBER)
        # 功能牌
        for color in [c for c in CardColor if c != CardColor.BLACK]:
            for _ in range(2):
                add_card(color, "skip", CardType.SKIP)
                add_card(color, "reverse", CardType.REVERSE)
                add_card(color, "draw_two", CardType.DRAW_TWO)
        # 万能牌
        for _ in range(4):
            add_card(CardColor.BLACK, "wild", CardType.WILD)
            add_card(CardColor.BLACK, "wild_draw_four", CardType.WILD_DRAW_FOUR)
        return cards

    def initialize_deck(self):
        """初始化牌堆"""
        self.deck = [*self.all_cards]
        random.shuffle(self.deck)

    def reset(self):
        """重置游戏"""
        self.players = [Player() for _ in range(self.num_players)]
        self.initialize_deck()
        self.discard_pile = []
        self.current_player_idx = 0
        self.direction = 1

        # 发牌
        for player in self.players:
            player.hand = [self.deck.pop() for _ in range(7)]

        # 初始化弃牌堆
        while True:
            top_card = self.deck.pop()
            if top_card.type not in [CardType.WILD, CardType.WILD_DRAW_FOUR]:
                self.discard_pile.append(top_card)
                self.current_color = top_card.color
                break
            else:
                self.deck.insert(0, top_card)

        return self._get_observation()

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        if len(self.players) == 0:
            raise ValueError("Game has not been initialized. Call `reset()` first.")

        """执行动作"""
        player = self.players[self.current_player_idx]
        reward = 0
        done = False
        info = {"message": ""}

        if not self._is_valid_action(action, player):
            reward = -1
            info["message"] = "Invalid action"
            return self._get_observation(), reward, done, info

        if action < 108:
            card = self.all_cards[action]
            self._play_card(card, player)
            reward = 1
        else:
            player.hand.extend(self._draw_cards(1))
            reward = -0.1

        # 检查胜利条件
        if len(player.hand) == 0:
            reward = 100
            done = True
            info["message"] = "Player wins!"
            return self._get_observation(), reward, done, info

        # 转换玩家回合
        while True:
            self.current_player_idx = (
                self.current_player_idx + self.direction
            ) % self.num_players
            if self.players[self.current_player_idx].skip:
                self.players[self.current_player_idx].skip = False
            else:
                break

        return self._get_observation(), reward, done, info

    def _replenish_deck(self):
        """补充牌堆：当牌堆用尽时，用弃牌堆重新洗牌"""
        top_card = self.discard_pile[-1]
        self.deck = self.discard_pile[:-1]
        random.shuffle(self.deck)
        self.discard_pile = [top_card]

    def _is_valid_action(self, action: int, player: "Player") -> bool:
        """验证出牌是否合法"""
        top_card = self.discard_pile[-1]
        if not 0 <= action <= 108:
            return False

        if action == 108:
            return True

        card = self.all_cards[action]

        if card not in player.hand:
            return False

        if card.color == self.current_color:
            return True

        if card.value == top_card.value:
            return True

        if card.color == CardColor.BLACK:
            return True

    def _play_card(self, card: Card, player: "Player"):
        """处理出牌逻辑"""
        player.hand.remove(card)
        self.discard_pile.append(card)

        # 处理颜色变化
        if card.color == CardColor.BLACK:
            self.current_color = self._choose_wild_color()
        else:
            self.current_color = card.color

        card_type = card.type
        # 2人游戏时，反转牌等同于跳过牌
        if card.type == CardType.REVERSE and len(self.players) == 2:
            card_type = CardType.SKIP

        next_player = self.players[
            (self.current_player_idx + self.direction) % self.num_players
        ]
        # 处理特殊牌效果
        if card_type == CardType.SKIP:
            next_player.skip = True
        elif card_type == CardType.REVERSE:
            self.direction *= -1
        elif card_type == CardType.DRAW_TWO:
            next_player.hand.extend(self._draw_cards(2))
            next_player.skip = True
        elif card_type == CardType.WILD_DRAW_FOUR:
            next_player.hand.extend(self._draw_cards(4))
            next_player.skip = True

    def _choose_wild_color(self) -> CardColor:
        """万能牌颜色选择（简单实现：选择手牌中最多的颜色，实际应该让玩家选择）"""
        color_counts = {color: 0 for color in CardColor if color != CardColor.BLACK}
        for card in self.players[self.current_player_idx].hand:
            if card.color != CardColor.BLACK:
                color_counts[card.color] += 1
        return max(color_counts, key=color_counts.get)

    def _draw_cards(self, num_cards: int) -> List[Card]:
        """抽牌"""
        cards = []
        if len(self.deck) < num_cards:
            self._replenish_deck()
        for _ in range(num_cards):
            if len(self.deck) > 0:
                cards.append(self.deck.pop())
        return cards

    def _get_observation(self) -> dict:
        """获取观察状态"""
        return {
            "hand": [
                card.unique_id for card in self.players[self.current_player_idx].hand
            ],
            "current_color": self.current_color.value if self.current_color else None,
            "direction": 1 if self.direction == 1 else 0,
            "player_turn": self.current_player_idx,
            "discard_pile": [card.unique_id for card in self.discard_pile],
            "deck_size": len(self.deck),
        }

    def render(self, show_all=True):
        if len(self.players) == 0:
            raise ValueError("Game has not been initialized. Call `reset()` first.")
        if self.renderer is None:
            raise NotImplementedError("No renderer provided")
        self.renderer.render(self, show_all)

    def user_input(self, valid_actions: list) -> int:
        if len(self.players) == 0:
            raise ValueError("Game has not been initialized. Call `reset()` first.")
        if self.renderer is None:
            raise NotImplementedError("No renderer provided")
        return self.renderer.user_input(self, valid_actions)


# --------------------------
# 玩家类定义
# --------------------------


class Player:
    def __init__(self):
        self.hand: List[Card] = []
        self.skip = False


# --------------------------
# 渲染器
# --------------------------
class UNORenderer(ABC):
    @abstractmethod
    def render(self, env: UNOGame, show_all=True):
        pass

    def user_input(self, env: UNOGame, valid_actions: list) -> int:
        raise NotImplementedError("User input not implemented")
