package com.craftsentient.craftmind.utils;

public class PrintUtils {
    private static String BLACK = "\033[30m";
    private static String RED = "\033[31m";
    private static String GREEN = "\033[32m";
    private static String YELLOW = "\033[33m";
    private static String BLUE = "\033[34m";
    private static String PURPLE = "\033[35m";
    private static String CYAN = "\033[36m";
    private static String WHITE = "\033[37m";

    private static String BLACK_BACKGROUND = "\033[40m";
    private static String RED_BACKGROUND  = "\033[41m";
    private static String GREEN_BACKGROUND  = "\033[42m";
    private static String YELLOW_BACKGROUND  = "\033[43m";
    private static String BLUE_BACKGROUND  = "\033[44m";
    private static String PURPLE_BACKGROUND  = "\033[45m";
    private static String CYAN_BACKGROUND  = "\033[46m";
    private static String WHITE_BACKGROUND  = "\033[47m";

    private static String BOLD = "\033[1m";
    private static String RESET = "\033[0m";

    public static String black(String str){
        return BLACK + str + RESET;
    }
    public static String red(String str){
        return RED + str + RESET;
    }
    public static String green(String str){
        return GREEN + str + RESET;
    }
    public static String yellow(String str){
        return YELLOW + str + RESET;
    }
    public static String blue(String str){
        return BLUE + str + RESET;
    }
    public static String purple(String str){
        return PURPLE + str + RESET;
    }
    public static String cyan(String str){
        return CYAN + str + RESET;
    }
    public static String white(String str){
        return WHITE + str + RESET;
    }

    public static String backgroundBlack(String str){
        return BLACK_BACKGROUND + str + RESET;
    }
    public static String backgroundRed(String str){
        return RED_BACKGROUND + str + RESET;
    }
    public static String backgroundGreen(String str){
        return GREEN_BACKGROUND + str + RESET;
    }
    public static String backgroundYellow(String str){
        return YELLOW_BACKGROUND + str + RESET;
    }
    public static String backgroundBlue(String str){
        return BLUE_BACKGROUND + str + RESET;
    }
    public static String backgroundPurple(String str){
        return PURPLE_BACKGROUND + str + RESET;
    }
    public static String backgroundCyan(String str){
        return CYAN_BACKGROUND + str + RESET;
    }
    public static String backgroundWhite(String str){
        return WHITE_BACKGROUND + str + RESET;
    }

    public static String bold(String str){
        return BOLD + str + RESET;
    }


}
