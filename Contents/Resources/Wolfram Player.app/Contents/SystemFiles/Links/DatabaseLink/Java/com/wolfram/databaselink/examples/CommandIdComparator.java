package com.wolfram.databaselink.examples;

import java.util.Comparator;

public class CommandIdComparator implements Comparator
{
  public int compare(Object a, Object b)
  {
    Command commandA = (Command)a;
    Command commandB = (Command)b;
    try
    {
      return (new Integer(commandA.getId())).compareTo(new Integer(commandB.getId()));
    } catch (Exception e) {}
    return 0;
  }
}
