using System.Diagnostics;
using System.Numerics;

const int arraySize = 200_000_000;
const int fillValue = 2;
const int iterations = 251;

int expectedSum = arraySize * fillValue;

Console.WriteLine("Подготовка данных...");
Console.WriteLine($"Размер массива: {arraySize:N0}");
Console.WriteLine($"Количество итераций: {iterations}");
Console.WriteLine($"Ожидаемая сумма: {expectedSum:N0}");
Console.WriteLine($"Аппаратная поддержка SIMD (Vector<T>): {(Vector.IsHardwareAccelerated ? "Да" : "Нет")}");
Console.WriteLine($"Размер вектора <int>: {Vector<int>.Count} элементов\n");

int[] data = new int[arraySize];
Array.Fill(data, fillValue);

Console.WriteLine("Тесты запущены. Пожалуйста, подождите...\n");

await RunTestAsync("1. Ручное управление потоками (Thread)", () => Task.FromResult(SumUsingThreads(data)), expectedSum);
await RunTestAsync("2. Ручное управление потоками + SIMD", () => Task.FromResult(SumUsingThreadsAndSimd(data)), expectedSum);
await RunTestAsync("3. Task Parallel Library (TPL)", () => Task.FromResult(SumUsingTPL(data)), expectedSum);
await RunTestAsync("4. Task Parallel Library (TPL) + SIMD", () => Task.FromResult(SumUsingTplAndSimd(data)), expectedSum);
await RunTestAsync("5. Асинхронное бинарное дерево редукции", () => SumUsingBinaryReductionTree(data), expectedSum);
await RunTestAsync("6. Асинхронное бинарное дерево редукции + SIMD", () => SumUsingBinaryReductionTreeAndSimd(data), expectedSum);
await RunTestAsync("7. Асинхронное N-арное дерево + SIMD", () => SumUsingSimdReductionTree(data), expectedSum);
await RunTestAsync("8. Linq.Sum()", async () => await Task.Run(() => data.Sum()), expectedSum);
await RunTestAsync("9. Linq.AsParallel().Sum()", async () => await Task.Run(() => data.AsParallel().Sum()), expectedSum);

Console.WriteLine("\nВсе тесты завершены.");
Console.ReadLine();

static async Task RunTestAsync(string testName, Func<Task<int>> sumFunction, int expectedSum)
{
    Console.WriteLine($"--- Тестирование: {testName} ---");
    List<double> executionTimes = [];
    Stopwatch stopwatch = new Stopwatch();

    int firstResult = 0;
    for (int i = 0; i < iterations; i++)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();

        stopwatch.Restart();
        int result = await sumFunction();
        stopwatch.Stop();

        if (i == 0) firstResult = result;

        double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
        executionTimes.Add(elapsedMs);
    }

    if (firstResult != expectedSum)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"ОШИБКА! Результат {firstResult:N0}, ожидалось {expectedSum:N0}");
        Console.ResetColor();
    }

    Console.WriteLine($"  Минимальное время: {executionTimes.Min():F2} мс");
    Console.WriteLine($"  Среднее время:    {executionTimes.Average():F2} мс");
    Console.WriteLine($"  Медианное время:  {CalculateMedian(executionTimes):F2} мс\n");
}

static int SumUsingThreads(int[] data)
{
    int threadCount = Environment.ProcessorCount;
    int[] partialSums = new int[threadCount];
    List<Thread> threads = [];
    int chunkSize = data.Length / threadCount;

    for (int i = 0; i < threadCount; i++)
    {
        int threadIndex = i;
        Thread thread = new Thread(() =>
        {
            int localSum = 0;
            int start = threadIndex * chunkSize;
            int end = (threadIndex == threadCount - 1) ? data.Length : start + chunkSize;
            for (int j = start; j < end; j++)
            {
                localSum += data[j];
            }
            partialSums[threadIndex] = localSum;
        });
        threads.Add(thread);
        thread.Start();
    }

    foreach (var thread in threads)
    {
        thread.Join();
    }

    return (int)partialSums.Sum();
}

static int SumUsingThreadsAndSimd(int[] data)
{
    if (!Vector.IsHardwareAccelerated) return SumUsingThreads(data);

    int threadCount = Environment.ProcessorCount;
    int[] partialSums = new int[threadCount];
    List<Thread> threads = [];
    int chunkSize = data.Length / threadCount;

    for (int i = 0; i < threadCount; i++)
    {
        int threadIndex = i;
        Thread thread = new Thread(() =>
        {
            int start = threadIndex * chunkSize;
            int end = (threadIndex == threadCount - 1) ? data.Length : start + chunkSize;
            partialSums[threadIndex] = SumVectorized(data.AsSpan(start, end - start));
        });
        threads.Add(thread);
        thread.Start();
    }

    foreach (var thread in threads)
    {
        thread.Join();
    }

    return (int)partialSums.Sum();
}

static int SumUsingTPL(int[] data)
{
    int totalSum = 0;
    Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, data.Length),
        () => 0,
        (range, _, localSum) =>
        {
            for (int i = range.Item1; i < range.Item2; i++)
            {
                localSum += data[i];
            }
            return localSum;
        },
        (localSum) => Interlocked.Add(ref totalSum, localSum));
    return totalSum;
}

static int SumUsingTplAndSimd(int[] data)
{
    if (!Vector.IsHardwareAccelerated) return SumUsingTPL(data);

    int totalSum = 0;
    Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, data.Length),
        () => 0,
        (range, _, localSum) =>
        {
            localSum += SumVectorized(data.AsSpan(range.Item1, range.Item2 - range.Item1));
            return localSum;
        },
        (localSum) => Interlocked.Add(ref totalSum, localSum));
    return totalSum;
}

static async Task<int> SumUsingBinaryReductionTree(int[] data)
{
    const int chunkSize = 256 * 1024; // 1MB chunk
    var tasks = new List<Task<int>>(data.Length / chunkSize + 1);

    // Уровень листьев: создаем задачи для суммирования каждого чанка
    for (int i = 0; i < data.Length; i += chunkSize)
    {
        int start = i;
        int end = Math.Min(start + chunkSize, data.Length);
        tasks.Add(Task.Run(() =>
        {
            int localSum = 0;
            for (int j = start; j < end; j++)
            {
                localSum += data[j];
            }
            return localSum;
        }));
    }

    // Уровни редукции: сводим результаты попарно, пока не останется одна задача
    while (tasks.Count > 1)
    {
        var nextLevelTasks = new List<Task<int>>();
        for (int i = 0; i < tasks.Count; i += 2)
        {
            var task1 = tasks[i];
            if (i + 1 < tasks.Count)
            {
                var task2 = tasks[i + 1];
                // Создаем новую задачу, которая ждет две дочерние и складывает их результаты
                nextLevelTasks.Add(Task.Run(async () => await task1 + await task2));
            }
            else
            {
                // Если количество задач нечетное, последняя просто переходит на следующий уровень
                nextLevelTasks.Add(task1);
            }
        }
        tasks = nextLevelTasks;
    }

    return await tasks[0];
}

static async Task<int> SumUsingBinaryReductionTreeAndSimd(int[] data)
{
    if (!Vector.IsHardwareAccelerated) return await SumUsingBinaryReductionTree(data);

    const int chunkSize = 256 * 1024; // 1MB chunk
    var tasks = new List<Task<int>>(data.Length / chunkSize + 1);

    // Уровень листьев: используем SIMD для суммирования каждого чанка
    for (int i = 0; i < data.Length; i += chunkSize)
    {
        int start = i;
        int end = Math.Min(start + chunkSize, data.Length);
        tasks.Add(Task.Run(() => (int)SumVectorized(data.AsSpan(start, end - start))));
    }

    // Уровни редукции (аналогично версии без SIMD)
    while (tasks.Count > 1)
    {
        var nextLevelTasks = new List<Task<int>>();
        for (int i = 0; i < tasks.Count; i += 2)
        {
            var task1 = tasks[i];
            if (i + 1 < tasks.Count)
            {
                var task2 = tasks[i + 1];
                nextLevelTasks.Add(Task.Run(async () => await task1 + await task2));
            }
            else
            {
                nextLevelTasks.Add(task1);
            }
        }
        tasks = nextLevelTasks;
    }

    return await tasks[0];
}

static async Task<int> SumUsingSimdReductionTree(int[] data)
{
    if (!Vector.IsHardwareAccelerated) return SumUsingTPL(data);

    const int chunkSize = 256 * 1024;
    var groupSize = Environment.ProcessorCount;

    var tasks = new List<Task<int>>(data.Length / chunkSize + 1);

    for (int i = 0; i < data.Length; i += chunkSize)
    {
        int start = i;
        int end = Math.Min(start + chunkSize, data.Length);
        tasks.Add(Task.Run(() => SumVectorized(data.AsSpan(start, end - start))));
    }

    while (tasks.Count > 1)
    {
        var nextLevelTasks = new List<Task<int>>();
        for (int i = 0; i < tasks.Count; i += groupSize)
        {
            var tasksInGroup = tasks.Skip(i).Take(groupSize).ToArray();
            nextLevelTasks.Add(Task.Run(async () =>
            {
                var partialSums = await Task.WhenAll(tasksInGroup);
                return SumVectorized(partialSums);
            }));
        }
        tasks = nextLevelTasks;
    }

    return await tasks[0];
}

static int SumVectorized(ReadOnlySpan<int> data)
{
    int i = 0;
    var vectorSize = Vector<int>.Count;
    var vectorSum = Vector<int>.Zero;
    var end = data.Length - (data.Length % vectorSize);

    for (; i < end; i += vectorSize)
    {
        vectorSum += new Vector<int>(data.Slice(i, vectorSize));
    }

    int scalarSum = Vector.Dot(vectorSum, Vector<int>.One);

    for (; i < data.Length; i++)
    {
        scalarSum += data[i];
    }

    return scalarSum;
}

static double CalculateMedian(List<double> numbers)
{
    if (numbers.Count == 0) return 0;
    var sortedNumbers = numbers.OrderBy(n => n).ToList();
    int mid = sortedNumbers.Count / 2;
    return (sortedNumbers.Count % 2 != 0) ? sortedNumbers[mid] : (sortedNumbers[mid - 1] + sortedNumbers[mid]) / 2.0;
}
