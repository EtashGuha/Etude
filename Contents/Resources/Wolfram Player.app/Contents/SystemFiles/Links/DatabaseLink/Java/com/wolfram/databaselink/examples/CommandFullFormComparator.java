package com.wolfram.databaselink.examples;

import java.util.Comparator;

public class CommandFullFormComparator implements Comparator
{
  public int compare(Object a, Object b)
  {
    Command commandA = (Command)a;
    Command commandB = (Command)b;
    return commandA.getFullForm().compareToIgnoreCase(commandB.getFullForm());
  }
}

