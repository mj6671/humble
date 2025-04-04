-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Mar 09, 2025 at 07:59 AM
-- Server version: 10.4.25-MariaDB
-- PHP Version: 7.4.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `pytrade`
--

-- --------------------------------------------------------

--
-- Table structure for table `binance_orders`
--

CREATE TABLE `binance_orders` (
  `id` int(11) NOT NULL,
  `users_order_id` int(11) DEFAULT NULL,
  `user_id` int(11) DEFAULT NULL,
  `binance_order_id` int(11) DEFAULT NULL,
  `symbol` varchar(50) DEFAULT NULL,
  `side` varchar(10) DEFAULT NULL,
  `price` decimal(18,8) DEFAULT NULL,
  `quantity` decimal(18,8) DEFAULT NULL,
  `status` varchar(50) DEFAULT NULL,
  `entry_conditions` text DEFAULT NULL,
  `date_added` datetime NOT NULL DEFAULT current_timestamp(),
  `date_updated` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `users_api`
--

CREATE TABLE `users_api` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `api_key` text DEFAULT NULL,
  `secret_key` text DEFAULT NULL,
  `exchange` varchar(20) NOT NULL,
  `status` tinyint(4) NOT NULL DEFAULT 0,
  `date_added` datetime NOT NULL DEFAULT current_timestamp(),
  `date_updated` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `users_api`
--

INSERT INTO `users_api` (`id`, `user_id`, `api_key`, `secret_key`, `exchange`, `status`, `date_added`, `date_updated`) VALUES
(1, 1, 'lauG5INzyInsCfUUeX7HqcJbHwvRNxYDhZhE1nOn9vPiVdc2yjiJ2y2uesqWvtA4', 'tgjsJiHgLhReJDqp0LQ4AFYZ0YcssTsIk0PlgPPQ3dbPJw0mPqRXLHBfqTXQTnLN', 'binance', 1, '2025-03-07 17:00:36', '2025-03-07 21:52:16');

-- --------------------------------------------------------

--
-- Table structure for table `users_orders`
--

CREATE TABLE `users_orders` (
  `order_id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `order_type` varchar(10) DEFAULT NULL,
  `currency_symbol` varchar(50) DEFAULT NULL,
  `price` decimal(18,8) DEFAULT NULL,
  `quantity` decimal(18,8) DEFAULT NULL,
  `status` varchar(50) DEFAULT NULL,
  `date_added` datetime NOT NULL DEFAULT current_timestamp(),
  `date_updated` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `users_orders`
--

INSERT INTO `users_orders` (`order_id`, `user_id`, `order_type`, `currency_symbol`, `price`, `quantity`, `status`, `date_added`, `date_updated`) VALUES
(1, 1, 'BUY', 'ETHUSDT', '2186.49000000', '0.01000000', 'PENDING', '2025-03-07 18:12:06', '2025-03-08 21:19:06'),
(2, 1, 'SELL', 'ETHUSDT', '2186.49000000', '0.01000000', 'PENDING', '2025-03-07 18:12:06', '2025-03-08 21:19:09');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `binance_orders`
--
ALTER TABLE `binance_orders`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users_api`
--
ALTER TABLE `users_api`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users_orders`
--
ALTER TABLE `users_orders`
  ADD PRIMARY KEY (`order_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `binance_orders`
--
ALTER TABLE `binance_orders`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users_api`
--
ALTER TABLE `users_api`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `users_orders`
--
ALTER TABLE `users_orders`
  MODIFY `order_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
