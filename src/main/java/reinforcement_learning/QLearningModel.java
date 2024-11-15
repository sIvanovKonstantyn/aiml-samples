package reinforcement_learning;

import java.util.Arrays;
import java.util.Random;

public class QLearningModel {
    private static final int GRID_SIZE = 5;
    private static final int ACTIONS = 4;

    // Q-Learning Parameters
    private static final double LEARNING_RATE = 0.1;
    private static final double DISCOUNT_FACTOR = 0.9;
    private static final double EXPLORATION_RATE = 0.2; // Probability of a random action
    private static final int EPISODES = 500; // Training iterations #

    private static final double REWARD_GOAL = 100;
    private static final double REWARD_TRAP = -100;
    private static final double STEP_PENALTY = -1;

    private final double[][][] qTable;

    // Goal and traps locations
    private static final int[] GOAL = {4, 4};
    private static final int[][] TRAPS = {{2, 2}, {3, 3}};

    public static void main(String[] args) {
        QLearningModel rl = new QLearningModel();
        rl.train();
        System.out.println("Q Table: " + Arrays.deepToString(rl.qTable));

        rl.runSimulation();
    }

    public QLearningModel() {
        qTable = new double[GRID_SIZE][GRID_SIZE][ACTIONS];
    }

    public void train() {
        Random random = new Random();

        for (int episode = 0; episode < EPISODES; episode++) {
            int x = random.nextInt(GRID_SIZE);
            int y = random.nextInt(GRID_SIZE);

            while (!isGoal(x, y)) {
                int action = random.nextDouble() < EXPLORATION_RATE ? random.nextInt(ACTIONS) : bestAction(x, y);

                int[] newPosition = move(x, y, action);
                int newX = newPosition[0];
                int newY = newPosition[1];

                double reward = getReward(newX, newY);

                double maxFutureQ = maxQ(newX, newY);
                qTable[x][y][action] = qTable[x][y][action] +
                        LEARNING_RATE * (reward + DISCOUNT_FACTOR * maxFutureQ - qTable[x][y][action]);

                x = newX;
                y = newY;
            }
        }

        System.out.println("Training is finished.");
    }

    public void runSimulation() {
        int x = 0, y = 0;

        System.out.println("Simulation started: Agent location: (0,0)");

        while (!isGoal(x, y)) {
            int action = bestAction(x, y);
            int[] newPosition = move(x, y, action);
            x = newPosition[0];
            y = newPosition[1];

            System.out.println("Agent moved to (" + x + "," + y + ")");
        }

        System.out.println("Agent archived the destination!");
    }

    private double getReward(int x, int y) {
        if (isGoal(x, y)) {
            return REWARD_GOAL;
        }
        for (int[] trap : TRAPS) {
            if (trap[0] == x && trap[1] == y) {
                return REWARD_TRAP;
            }
        }
        return STEP_PENALTY;
    }

    private boolean isGoal(int x, int y) {
        return GOAL[0] == x && GOAL[1] == y;
    }

    private int bestAction(int x, int y) {
        double maxQ = Double.NEGATIVE_INFINITY;
        int bestAction = 0;

        for (int a = 0; a < ACTIONS; a++) {
            if (qTable[x][y][a] > maxQ) {
                maxQ = qTable[x][y][a];
                bestAction = a;
            }
        }

        return bestAction;
    }

    private double maxQ(int x, int y) {
        double maxQ = Double.NEGATIVE_INFINITY;

        for (int a = 0; a < ACTIONS; a++) {
            if (qTable[x][y][a] > maxQ) {
                maxQ = qTable[x][y][a];
            }
        }

        return maxQ;
    }

    private int[] move(int x, int y, int action) {
        return switch (action) {
            case 0 -> // up
                    new int[]{Math.max(0, x - 1), y};
            case 1 -> // right
                    new int[]{x, Math.min(GRID_SIZE - 1, y + 1)};
            case 2 -> // down
                    new int[]{Math.min(GRID_SIZE - 1, x + 1), y};
            case 3 -> // left
                    new int[]{x, Math.max(0, y - 1)};
            default -> new int[]{x, y};
        };
    }
}
